from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import threading
import os
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque
import csv
import io

import torch

print("CUDA available:", torch.cuda.is_available())

# Flask 설정
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 전역 변수
current_frame = None
video_info = None
processing_thread = None
stop_processing = False
frame_data_log = []
vehicle_tracker = defaultdict(lambda: deque(maxlen=60))  # 차량 궤적 추적
next_vehicle_id = 0

# 진행률 전역 변수
processed_frames = 0
total_frames = 0

# YOLO 모델 로드 (처음 한 번만)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

model = YOLO('yolo11l.pt')   # 또는 'yolo11m'
model.to(device)
model.conf = 0.35
model.iou = 0.45

# Vehicle class 집합 (car, motorcycle, bus, truck 등)
VEHICLE_CLASSES = {2, 3, 5, 7}


class VehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = 100

    def update(self, detections):
        """
        detections: list of (x, y, w, h, conf, class_id)
        returns: list of (track_id, x, y, w, h, conf)
        """
        if not detections:
            return []

        # 이전 트랙의 중심
        current_centers = {}
        for track_id, (x, y, w, h, conf, cls_id) in self.tracks.items():
            current_centers[track_id] = (x + w / 2, y + h / 2)

        # 새 detections의 중심
        new_centers = {}
        for i, (x, y, w, h, conf, cls_id) in enumerate(detections):
            new_centers[i] = (x + w / 2, y + h / 2)

        # 최근접 이웃 매칭
        matched_tracks = set()
        new_tracks = {}

        for new_idx, new_center in new_centers.items():
            best_track_id = None
            best_distance = self.max_distance

            for track_id, old_center in current_centers.items():
                if track_id in matched_tracks:
                    continue
                dist = np.sqrt((new_center[0] - old_center[0]) ** 2 +
                               (new_center[1] - old_center[1]) ** 2)
                if dist < best_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                matched_tracks.add(best_track_id)
                new_tracks[best_track_id] = detections[new_idx]
            else:
                new_tracks[self.next_id] = detections[new_idx]
                self.next_id += 1

        self.tracks = new_tracks
        # cls_id는 바깥에선 안 쓰니까 리턴에는 5개만
        return [(tid, x, y, w, h, conf) for tid, (x, y, w, h, conf, cls_id) in new_tracks.items()]


def detect_lanes(frame):
    """차선/트랙 경계 검출"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 100, 200)

    # ROI 설정 (아래쪽 절반만 - 주행 방향)
    h, w = frame.shape[:2]
    roi_mask = np.zeros_like(edges)
    roi_mask[h // 3:, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    # Hough 라인 변환
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    left_line = None
    right_line = None

    if lines is not None:
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append((slope, line[0]))

        # 왼쪽(음의 기울기), 오른쪽(양의 기울기) 분류
        left_lines = [l for s, l in slopes if s < -0.3]
        right_lines = [l for s, l in slopes if s > 0.3]

        if left_lines:
            left_line = np.mean(left_lines, axis=0).astype(int)
        if right_lines:
            right_line = np.mean(right_lines, axis=0).astype(int)

    return left_line, right_line, edges


def detect_vehicles(frame):
    """YOLO를 이용한 차량 검출"""
    results = model(frame, verbose=False)

    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls)
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                detections.append((x, y, w, h, conf, cls_id))

    return detections


def calculate_speed(track_history, pixels_per_meter=15, fps=30):
    """픽셀 이동으로부터 속도 계산"""
    if len(track_history) < 2:
        return 0

    # 최근 10프레임의 이동 거리
    history_list = list(track_history)
    if len(history_list) >= 10:
        recent_positions = history_list[-10:]
    else:
        recent_positions = history_list

    x1, y1 = recent_positions[0]
    x2, y2 = recent_positions[-1]

    pixel_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist = pixel_dist / pixels_per_meter  # meters
    time_delta = len(recent_positions) / fps  # seconds

    speed_ms = real_dist / time_delta if time_delta > 0 else 0
    speed_kmh = speed_ms * 3.6

    return speed_kmh


def calculate_overtaking_score(ego_center, lead_center, ego_speed, lead_speed,
                               left_line, right_line, frame_h, frame_w):
    """추월 가능성 계산"""

    # 전방 거리 (픽셀 -> 미터)
    if lead_center and ego_center:
        dy = abs(lead_center[1] - ego_center[1])
        forward_dist = dy / 15 * 10  # 간단한 환산
    else:
        forward_dist = 100

    # 속도 차
    speed_delta = ego_speed - lead_speed

    # 옆 공간 여유 (프레임 폭 기준)
    lateral_margin = frame_w / 2  # 간단히 폭의 절반

    # 스코어 계산
    distance_score = max(0, 1 - (forward_dist / 100))
    speed_score = min(1, max(0, speed_delta / 50))
    space_score = max(0, min(1, lateral_margin / 50))

    total_score = (distance_score * 0.3 + speed_score * 0.5 + space_score * 0.2) * 100

    if total_score < 30:
        status = "NOT_POSSIBLE"
        emoji = "🔴"
    elif total_score < 70:
        status = "RISKY"
        emoji = "🟡"
    else:
        status = "POSSIBLE"
        emoji = "🟢"

    return total_score, status, emoji


def process_video(video_path):
    global current_frame, stop_processing, frame_data_log, processed_frames, total_frames

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    tracker = VehicleTracker()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frame_data_log = []
    processed_frames = 0
    frame_skip = 1  # 전체 프레임의 절반만 처리

    while not stop_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 스킵
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # 크기 조정
        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]

        # 차선 검출
        left_line, right_line, edges = detect_lanes(frame)

        # 차량 검출 및 추적
        detections = detect_vehicles(frame)
        tracked = tracker.update(detections)

        # 분석 결과 그리기
        result_frame = frame.copy()

        # 차선 그리기
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 차량 박스 및 궤적 그리기
        ego_car = None
        lead_car = None

        for track_id, x, y, w_box, h_box, conf in tracked:
            center = (int(x + w_box / 2), int(y + h_box / 2))
            vehicle_tracker[track_id].append(center)

            # 궤적 그리기
            points = list(vehicle_tracker[track_id])
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(result_frame, points[i], points[i + 1], (0, 255, 255), 1)

            # 박스 그리기
            cv2.rectangle(result_frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            cv2.putText(result_frame, f"ID:{track_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 자아 차량(프레임 하단 중앙) vs 앞차(더 위)
            if ego_car is None or abs(center[0] - w // 2) < abs(ego_car[1][0] - w // 2):
                ego_car = (track_id, center)

        # 추월 분석
        overtake_score = 0
        overtake_status = "UNKNOWN"
        overtake_emoji = "⚪"

        if ego_car and len(tracked) > 1:
            ego_id, ego_center = ego_car
            ego_speed = calculate_speed(vehicle_tracker[ego_id])

            # 앞차 찾기
            for track_id, x, y, w_box, h_box, conf in tracked:
                if track_id != ego_id:
                    center = (int(x + w_box / 2), int(y + h_box / 2))
                    if center[1] < ego_center[1]:  # 앞에 있음
                        lead_car = (track_id, center)
                        break

        # 앞차와 내차 사이 거리 라인 그리기
        if ego_car and lead_car:
            ego_id, ego_center = ego_car
            lead_id, lead_center = lead_car

            # 픽셀 거리를 미터로 환산
            pixel_dist = np.sqrt((lead_center[0] - ego_center[0]) ** 2 +
                                 (lead_center[1] - ego_center[1]) ** 2)
            distance_m = pixel_dist / 15

            # 거리에 따라 라인 색상 결정
            if distance_m < 10:
                line_color = (0, 0, 255)  # 빨강 (위험)
            elif distance_m < 20:
                line_color = (0, 255, 255)  # 노랑 (주의)
            else:
                line_color = (0, 255, 0)  # 초록 (안전)

            # 라인 그리기
            cv2.line(result_frame, ego_center, lead_center, line_color, 3)

            # 중점에 거리 표시
            mid_x = (ego_center[0] + lead_center[0]) // 2
            mid_y = (ego_center[1] + lead_center[1]) // 2
            cv2.putText(result_frame, f"{distance_m:.1f}m", (mid_x - 20, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

        # 추월 점수 계산
        if ego_car and lead_car:
            ego_id, ego_center = ego_car
            lead_id, lead_center = lead_car
            ego_speed = calculate_speed(vehicle_tracker[ego_id])
            lead_speed = calculate_speed(vehicle_tracker[lead_id])

            overtake_score, overtake_status, overtake_emoji = calculate_overtaking_score(
                ego_center, lead_center, ego_speed, lead_speed,
                left_line, right_line, h, w
            )

        # 대시보드 오버레이
        # 자아 차 정보
        if ego_car:
            ego_id, ego_center = ego_car
            ego_speed = calculate_speed(vehicle_tracker[ego_id])
            cv2.putText(result_frame, f"EGO SPEED: {ego_speed:.1f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 앞차 정보
        if lead_car:
            lead_id, lead_center = lead_car
            lead_speed = calculate_speed(vehicle_tracker[lead_id])
            cv2.putText(result_frame, f"LEAD SPEED: {lead_speed:.1f} km/h", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 추월 상태
        cv2.putText(result_frame, f"OVERTAKING: {overtake_emoji} {overtake_status} ({overtake_score:.1f}%)",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 프레임 정보
        cv2.putText(result_frame, f"Frame: {frame_count}", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        current_frame = result_frame
        processed_frames = frame_count  # 진행률용

        # 데이터 로깅
        log_entry = {
            'frame': frame_count,
            'timestamp': frame_count / fps if fps > 0 else 0,
            'ego_speed': calculate_speed(vehicle_tracker[ego_car[0]]) if ego_car else 0,
            'lead_speed': calculate_speed(vehicle_tracker[lead_car[0]]) if lead_car else 0,
            'overtake_score': overtake_score,
            'overtake_status': overtake_status
        }
        frame_data_log.append(log_entry)

        frame_count += 1

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global processing_thread, stop_processing, video_info, total_frames, processed_frames

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.mp4'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 비디오 정보
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        video_info = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0,
            'height': h,
            'width': w
        }

        total_frames = frame_count
        processed_frames = 0

        # 백그라운드에서 처리 시작
        stop_processing = False
        processing_thread = threading.Thread(target=process_video, args=(filepath,))
        processing_thread.start()

        return jsonify({'status': 'processing', 'video_info': video_info})

    return jsonify({'error': 'Invalid file'}), 400


@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        while True:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')
            else:
                # 프레임 없을 때 잠깐 대기
                import time
                time.sleep(0.03)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/progress')
def progress():
    global processed_frames, total_frames
    if total_frames == 0:
        return jsonify({'progress': 0, 'processed_frames': 0, 'total_frames': 0})
    ratio = min(100, max(0, processed_frames / total_frames * 100))
    return jsonify({
        'progress': ratio,
        'processed_frames': processed_frames,
        'total_frames': total_frames
    })


@app.route('/telemetry')
def telemetry():
    global frame_data_log, video_info, processed_frames, total_frames

    if not frame_data_log:
        return jsonify({
            'ego_speed': None,
            'lead_speed': None,
            'speed_delta': None,
            'overtake_score': None,
            'overtake_status': 'READY',
            'fps': video_info['fps'] if video_info else None,
            'frame': processed_frames,
            'progress': (processed_frames / total_frames * 100) if total_frames else 0
        })

    last = frame_data_log[-1]
    ego_speed = last['ego_speed']
    lead_speed = last['lead_speed']
    speed_delta = ego_speed - lead_speed if (ego_speed is not None and lead_speed is not None) else None

    return jsonify({
        'ego_speed': ego_speed,
        'lead_speed': lead_speed,
        'speed_delta': speed_delta,
        'overtake_score': last['overtake_score'],
        'overtake_status': last['overtake_status'],
        'fps': video_info['fps'] if video_info else None,
        'frame': last['frame'],
        'progress': (processed_frames / total_frames * 100) if total_frames else 0
    })


@app.route('/download_data')
def download_data():
    global frame_data_log

    if not frame_data_log:
        return jsonify({'error': 'No data'}), 400

    # CSV로 변환
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['frame', 'timestamp', 'ego_speed', 'lead_speed', 'overtake_score',
                                                'overtake_status'])
    writer.writeheader()
    writer.writerows(frame_data_log)

    output.seek(0)
    bytes_io = io.BytesIO(output.getvalue().encode())

    return send_file(
        bytes_io,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'racing_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)