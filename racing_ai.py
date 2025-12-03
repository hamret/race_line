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

# Norfair 임포트
try:
    from norfair import Detection, Tracker

    NORFAIR_AVAILABLE = True
except ImportError:
    print("Warning: Norfair not installed. Install with: pip install norfair")
    NORFAIR_AVAILABLE = False

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
model = YOLO('yolo11x.pt')
model.to(device)

# ===== 개선: 신뢰도 임계값 최적화 =====
model.conf = 0.2  # 멀리 있는 작은 차도 탐지
model.iou = 0.4  # NMS 더 엄격하게

# F1: car 클래스만 사용
VEHICLE_CLASSES = {2, 3, 5, 7}

# 박스 크기 필터
MIN_BOX_W, MIN_BOX_H = 8, 8


def detect_lanes(frame):
    """차선/트랙 경계 검출"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 100, 200)

    h, w = frame.shape[:2]
    roi_mask = np.zeros_like(edges)
    roi_mask[h // 3:, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 50,
        minLineLength=50, maxLineGap=10
    )

    left_line = None
    right_line = None

    if lines is not None:
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append((slope, line[0]))

        left_lines = [l for s, l in slopes if s < -0.3]
        right_lines = [l for s, l in slopes if s > 0.3]

        if left_lines:
            left_line = np.mean(left_lines, axis=0).astype(int)
        if right_lines:
            right_line = np.mean(right_lines, axis=0).astype(int)

    return left_line, right_line, edges


def detect_vehicles(frame, min_confidence=0.2):
    """개선된 YOLO 차량 검출"""
    results = model(frame, verbose=False, conf=min_confidence)
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

                if w < MIN_BOX_W or h < MIN_BOX_H:
                    continue

                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5:
                    continue

                detections.append((x, y, w, h, conf, cls_id))

    return detections


def convert_to_norfair_detections(yolo_detections):
    """YOLO 검출을 Norfair Detection으로 변환"""
    norfair_detections = []
    for (x, y, w_box, h_box, conf, cls_id) in yolo_detections:
        cx = x + w_box / 2
        cy = y + h_box / 2
        points = np.array([[cx, cy]])
        scores = np.array([conf])
        norfair_detections.append(
            Detection(points=points, scores=scores,
                      data={"bbox": (x, y, w_box, h_box), "conf": conf})
        )
    return norfair_detections


def calculate_speed(track_history, pixels_per_meter=15, fps=30):
    """픽셀 이동으로부터 속도 계산"""
    if len(track_history) < 2:
        return 0

    history_list = list(track_history)
    if len(history_list) >= 10:
        recent_positions = history_list[-10:]
    else:
        recent_positions = history_list

    x1, y1 = recent_positions[0]
    x2, y2 = recent_positions[-1]

    pixel_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist = pixel_dist / pixels_per_meter
    time_delta = len(recent_positions) / fps
    speed_ms = real_dist / time_delta if time_delta > 0 else 0
    speed_kmh = speed_ms * 3.6

    return speed_kmh


def calculate_distance(ego_center, lead_center, frame_height=540):
    """개선된 거리 계산"""
    if not ego_center or not lead_center:
        return 0

    pixel_dist = np.sqrt(
        (lead_center[0] - ego_center[0]) ** 2 +
        (lead_center[1] - ego_center[1]) ** 2
    )

    pixels_per_meter = frame_height / 50
    distance_m = pixel_dist / pixels_per_meter

    return max(0, distance_m)


def calculate_overtaking_score(ego_center, lead_center,
                               ego_speed, lead_speed,
                               distance_m,
                               rel_dist,
                               left_line, right_line,
                               frame_h, frame_w,
                               recent_distances=None,
                               recent_speed_deltas=None,
                               recent_rels=None):
    """
    헤드캠 기준 개선된 추월 가능성 계산
    - rel_dist: 화면 비율 기반 상대 거리 (0~1)
    - recent_rels: rel_dist 추세
    """

    # === OVERTAKING 상태 처리: rel_dist < 0.05면 이미 추월 중 ===
    if rel_dist < 0.05:
        return 100.0, "OVERTAKING", "✅"

    # 1. 절대 거리 점수 (픽셀 기반, 참고용)
    if distance_m < 5:
        abs_dist_score = 0.9
    elif distance_m < 15:
        abs_dist_score = 0.7
    elif distance_m < 30:
        abs_dist_score = 0.4
    else:
        abs_dist_score = 0.1

    # 2. 상대 거리 점수 (헤드캠 기준 - 더 중요함)
    if rel_dist < 0.10:
        rel_score = 0.95
    elif rel_dist < 0.20:
        rel_score = 0.85
    elif rel_dist < 0.35:
        rel_score = 0.65
    else:
        rel_score = 0.30

    # 3. 상대 거리 추세 (가까워지고 있는지)
    rel_trend_score = 0.5
    if recent_rels is not None and len(recent_rels) >= 3:
        old_rel = recent_rels[0]
        new_rel = recent_rels[-1]
        delta_rel = old_rel - new_rel

        norm_delta_rel = max(-1.0, min(1.0, delta_rel / 0.2))
        rel_trend_score = 0.5 + 0.5 * norm_delta_rel

    # 4. 거리 변화 기반 점수 (절대 거리 추세)
    dist_trend_score = 0.5
    if recent_distances is not None and len(recent_distances) >= 3:
        d_old = recent_distances[0]
        d_new = recent_distances[-1]
        delta_d = d_old - d_new

        norm_delta_d = max(-1.0, min(1.0, delta_d / 20.0))
        dist_trend_score = 0.5 + 0.5 * norm_delta_d

    # 5. 속도 차 점수
    speed_trend_score = 0.5
    if recent_speed_deltas is not None and len(recent_speed_deltas) > 0:
        avg_delta_v = sum(recent_speed_deltas) / len(recent_speed_deltas)
    else:
        avg_delta_v = ego_speed - lead_speed if (ego_speed is not None and lead_speed is not None) else 0.0

    if avg_delta_v <= 0:
        speed_trend_score = 0.1
    elif avg_delta_v < 5:
        speed_trend_score = 0.4
    elif avg_delta_v < 15:
        speed_trend_score = 0.7
    else:
        speed_trend_score = 0.9

    # 6. 옆 공간 여유도
    if ego_center and left_line is not None and right_line is not None:
        left_x = left_line[0] if len(left_line) > 0 else 0
        right_x = right_line[0] if len(right_line) > 0 else frame_w
        lane_width = abs(right_x - left_x) if right_x != left_x else frame_w
        margin = lane_width / 3
        space_score = max(0.2, min(1.0, margin / 100.0))
    else:
        space_score = 0.5

    # 7. 최종 점수 계산 (헤드캠 기준으로 상대 거리 비중 크게)
    total_score = (
                          rel_score * 0.45 +
                          rel_trend_score * 0.25 +
                          speed_trend_score * 0.2 +
                          space_score * 0.1
                  ) * 100.0

    # 8. 상태 구간 (헤드캠 기준 재조정)
    if total_score < 35:
        status = "DANGEROUS"
        emoji = "🔴"
    elif total_score < 55:
        status = "RISKY"
        emoji = "🟡"
    elif total_score < 75:
        status = "CAUTION"
        emoji = "🟠"
    else:
        status = "POSSIBLE"
        emoji = "🟢"

    return total_score, status, emoji


def process_video(video_path):
    global current_frame, stop_processing, frame_data_log, processed_frames, total_frames

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_data_log = []
    processed_frames = 0

    frame_skip = 1

    # Norfair 트래커 초기화
    if NORFAIR_AVAILABLE:
        mot_tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=30
        )
    else:
        print("Norfair not available, using frame_count as track_id")
        mot_tracker = None

    recent_lead_ids = deque(maxlen=2)
    recent_distances = deque(maxlen=15)
    recent_speed_deltas = deque(maxlen=15)
    recent_rels = deque(maxlen=15)

    last_lead_visible_frame = -100
    LEAD_VISIBILITY_GRACE = 30

    last_distance_m = 0.0
    last_rel_dist = 0.0
    last_ego_speed = 0.0
    last_lead_speed = 0.0
    last_overtake_score = 0.0
    last_overtake_status = "UNKNOWN"
    last_overtake_emoji = "⚪"

    while not stop_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]

        left_line, right_line, edges = detect_lanes(frame)
        detections = detect_vehicles(frame)

        result_frame = frame.copy()

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Norfair 트래킹
        if NORFAIR_AVAILABLE and mot_tracker is not None:
            norfair_detections = convert_to_norfair_detections(detections)
            tracked_objects = mot_tracker.update(detections=norfair_detections)
        else:
            # Fallback: detections를 그대로 사용
            tracked_objects = []
            for idx, (x, y, w_box, h_box, conf, cls_id) in enumerate(detections):
                class MockTrack:
                    def __init__(self, track_id, x, y, w_box, h_box, conf):
                        self.id = track_id
                        self.points = np.array([[x + w_box / 2, y + h_box / 2]])
                        self.last_detection = type('obj', (object,), {
                            'data': {"bbox": (x, y, w_box, h_box), "conf": conf},
                            'scores': np.array([conf])
                        })()

                tracked_objects.append(MockTrack(idx, x, y, w_box, h_box, conf))

        ego_car = None
        lead_car = None

        for trk in tracked_objects:
            track_id = trk.id

            # 중심 좌표 계산
            if hasattr(trk, 'estimate') and trk.estimate is not None:
                cx, cy = trk.estimate[0]
            else:
                cx, cy = trk.points[0]

            cx, cy = int(cx), int(cy)
            center = (cx, cy)

            # YOLO bbox 가져오기
            x, y, w_box, h_box = trk.last_detection.data["bbox"]

            vehicle_tracker[track_id].append(center)

            # 궤적 그리기
            points = list(vehicle_tracker[track_id])
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(result_frame, points[i], points[i + 1], (0, 255, 255), 1)

            # 박스 그리기
            cv2.rectangle(result_frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

            # 신뢰도
            confidence = float(trk.last_detection.scores.mean())
            cv2.putText(result_frame, f"ID:{track_id} ({confidence:.2f})", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # ego 차량 선택
            if center[1] > h * 0.6:
                if ego_car is None or abs(center[0] - w // 2) < abs(ego_car[1][0] - w // 2):
                    ego_car = (track_id, center)

        # lead 후보 찾기
        lead_candidates = []
        if ego_car:
            ego_id, ego_center = ego_car

            for trk in tracked_objects:
                track_id = trk.id
                if track_id != ego_id:
                    if hasattr(trk, 'estimate') and trk.estimate is not None:
                        cx, cy = trk.estimate[0]
                    else:
                        cx, cy = trk.points[0]

                    center = (int(cx), int(cy))
                    x, y, w_box, h_box = trk.last_detection.data["bbox"]

                    if center[1] < ego_center[1]:
                        if 0.2 * w < center[0] < 0.8 * w:
                            dy = ego_center[1] - center[1]
                            lead_candidates.append((track_id, center, dy))

            if lead_candidates:
                lead_candidates.sort(key=lambda x: x[2])
                best_lead_id = lead_candidates[0][0]

                if recent_lead_ids and lead_candidates[0][0] in recent_lead_ids:
                    best_lead_id = lead_candidates[0][0]
                else:
                    best_lead_id = lead_candidates[0][0]

                for track_id, center, dy in lead_candidates:
                    if track_id == best_lead_id:
                        lead_car = (track_id, center)
                        recent_lead_ids.append(track_id)
                        break

        distance_m = last_distance_m
        rel_dist = last_rel_dist
        ego_speed = last_ego_speed
        lead_speed = last_lead_speed
        overtake_score = last_overtake_score
        overtake_status = last_overtake_status
        overtake_emoji = last_overtake_emoji

        lead_visible = ego_car is not None and lead_car is not None

        if lead_visible:
            ego_id, ego_center = ego_car
            lead_id, lead_center = lead_car

            distance_m = calculate_distance(ego_center, lead_center, h)
            recent_distances.append(distance_m)

            rel_dist = (ego_center[1] - lead_center[1]) / h
            rel_dist = max(0.0, min(1.0, rel_dist))
            recent_rels.append(rel_dist)

            ego_speed = calculate_speed(vehicle_tracker[ego_id])
            lead_speed = calculate_speed(vehicle_tracker[lead_id])
            speed_delta = ego_speed - lead_speed
            recent_speed_deltas.append(speed_delta)

            overtake_score, overtake_status, overtake_emoji = calculate_overtaking_score(
                ego_center, lead_center,
                ego_speed, lead_speed,
                distance_m,
                rel_dist,
                left_line, right_line,
                h, w,
                recent_distances=list(recent_distances),
                recent_speed_deltas=list(recent_speed_deltas),
                recent_rels=list(recent_rels)
            )

            if distance_m < 10:
                line_color = (0, 0, 255)
            elif distance_m < 20:
                line_color = (0, 165, 255)
            else:
                line_color = (0, 255, 0)

            cv2.line(result_frame, ego_center, lead_center, line_color, 3)

            mid_x = (ego_center[0] + lead_center[0]) // 2
            mid_y = (ego_center[1] + lead_center[1]) // 2
            cv2.putText(result_frame, f"{distance_m:.1f}m ({rel_dist:.2f})", (mid_x - 40, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

            last_lead_visible_frame = frame_count
            last_distance_m = distance_m
            last_rel_dist = rel_dist
            last_ego_speed = ego_speed
            last_lead_speed = lead_speed
            last_overtake_score = overtake_score
            last_overtake_status = overtake_status
            last_overtake_emoji = overtake_emoji

        else:
            if frame_count - last_lead_visible_frame > LEAD_VISIBILITY_GRACE:
                distance_m = 0.0
                rel_dist = 0.0
                ego_speed = 0.0
                lead_speed = 0.0
                overtake_score = 0.0
                overtake_status = "UNKNOWN"
                overtake_emoji = "⚪"
                last_distance_m = distance_m
                last_rel_dist = rel_dist
                last_ego_speed = ego_speed
                last_lead_speed = lead_speed
                last_overtake_score = overtake_score
                last_overtake_status = overtake_status
                last_overtake_emoji = overtake_emoji

        if ego_car:
            ego_id, ego_center = ego_car
            ego_speed_display = calculate_speed(vehicle_tracker[ego_id])
            cv2.putText(result_frame, f"EGO SPEED: {ego_speed_display:.1f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if lead_car and lead_visible:
            lead_id, lead_center = lead_car
            lead_speed_display = calculate_speed(vehicle_tracker[lead_id])
            cv2.putText(result_frame,
                        f"LEAD SPEED: {lead_speed_display:.1f} km/h",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(result_frame,
                    f"OVERTAKING: {overtake_emoji} {overtake_status} ({overtake_score:.1f}%)",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        cv2.putText(result_frame,
                    f"Frame: {frame_count} | RelDist: {rel_dist:.2f}",
                    (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        current_frame = result_frame
        processed_frames = frame_count

        log_entry = {
            'frame': frame_count,
            'timestamp': frame_count / fps if fps > 0 else 0,
            'ego_speed': ego_speed,
            'lead_speed': lead_speed,
            'distance': distance_m,
            'rel_dist': rel_dist,
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
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')
            else:
                import time
                time.sleep(0.03)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
            'distance': None,
            'rel_dist': None,
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
    distance = last['distance']
    rel_dist = last.get('rel_dist', 0.0)

    return jsonify({
        'ego_speed': ego_speed,
        'lead_speed': lead_speed,
        'speed_delta': speed_delta,
        'distance': distance,
        'rel_dist': rel_dist,
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

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'frame', 'timestamp', 'ego_speed', 'lead_speed',
        'distance', 'rel_dist', 'overtake_score', 'overtake_status'
    ])
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
