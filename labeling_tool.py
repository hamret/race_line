import cv2
import os
from pathlib import Path


class YOLOLabeler:
    def __init__(self, image_folder, output_folder):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.image_files = sorted([f for f in os.listdir(image_folder)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.current_idx = 0
        self.boxes = []  # [(x1, y1, x2, y2), ...]
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.current_image = None
        self.original_image = None

        os.makedirs(output_folder, exist_ok=True)

    def mouse_callback(self, event, x, y, flags, param):
        if self.original_image is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 드래그 중: 초록색 임시 박스 + 이미 그린 박스(파란색) 같이 보여주기
                self.current_image = self.original_image.copy()
                cv2.rectangle(self.current_image,
                              (self.start_x, self.start_y), (x, y),
                              (0, 255, 0), 2)
                for box in self.boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(self.current_image, (x1, y1), (x2, y2),
                                  (255, 0, 0), 2)
                cv2.imshow('YOLO Labeler', self.current_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_x, self.start_y
                x2, y2 = x, y

                # 좌상단/우하단 정규화
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # 너무 작은 박스는 무시
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.boxes.append((x1, y1, x2, y2))
                    print(f"박스 추가: ({x1}, {y1}) ~ ({x2}, {y2})")

                # 최종 박스 상태 다시 그림
                self.current_image = self.original_image.copy()
                for box in self.boxes:
                    bx1, by1, bx2, by2 = box
                    cv2.rectangle(self.current_image, (bx1, by1), (bx2, by2),
                                  (255, 0, 0), 2)
                cv2.imshow('YOLO Labeler', self.current_image)

    def convert_to_yolo_format(self, image_shape):
        """
        YOLO format: class_id center_x center_y width height
        (0~1 정규화)
        """
        h, w = image_shape[:2]
        yolo_lines = []
        for x1, y1, x2, y2 in self.boxes:
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            # class_id = 0 (모든 차를 하나의 클래스로)
            yolo_lines.append(
                f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
            )
        return yolo_lines

    def save_annotations(self):
        """현재 이미지의 라벨을 YOLO format으로 저장"""
        if not self.image_files:
            return

        image_name = self.image_files[self.current_idx]
        label_name = Path(image_name).stem + '.txt'
        label_path = os.path.join(self.output_folder, label_name)

        if self.boxes:
            yolo_lines = self.convert_to_yolo_format(self.original_image.shape)
            with open(label_path, 'w', encoding='utf-8') as f:
                f.writelines(yolo_lines)
            print(f"저장됨: {label_path} ({len(self.boxes)}개 박스)")
        else:
            # 박스가 없으면 빈 파일 생성
            open(label_path, 'w', encoding='utf-8').close()
            print(f"저장됨: {label_path} (빈 파일)")

    def load_annotations(self):
        """이전에 저장된 라벨 로드"""
        if not self.image_files:
            return

        image_name = self.image_files[self.current_idx]
        label_name = Path(image_name).stem + '.txt'
        label_path = os.path.join(self.output_folder, label_name)

        self.boxes = []
        if os.path.exists(label_path):
            h, w = self.original_image.shape[:2]
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, center_x, center_y, width, height = map(float, parts)
                        # 정규화된 값 -> 픽셀 값으로 변환
                        x1 = int((center_x - width / 2) * w)
                        y1 = int((center_y - height / 2) * h)
                        x2 = int((center_x + width / 2) * w)
                        y2 = int((center_y + height / 2) * h)
                        self.boxes.append((x1, y1, x2, y2))
            print(f"로드됨: {label_path} ({len(self.boxes)}개 박스)")

    def display_image(self):
        """현재 이미지 표시"""
        if not self.image_files:
            print("이미지 폴더가 비어 있습니다!")
            return

        image_path = os.path.join(self.image_folder, self.image_files[self.current_idx])
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"이미지 로드 실패: {image_path}")
            return

        self.current_image = self.original_image.copy()

        self.load_annotations()

        # 로드된 박스 그리기
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)

        # 정보 표시
        h, w = self.original_image.shape[:2]
        info_text = f"Image {self.current_idx + 1}/{len(self.image_files)}: " \
                    f"{self.image_files[self.current_idx]} ({w}x{h})"
        cv2.putText(self.current_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLO Labeler', self.current_image)

    def run(self):
        """라벨링 UI 시작"""
        cv2.namedWindow('YOLO Labeler', cv2.WINDOW_NORMAL)  # 창 크기 조절 가능
        cv2.resizeWindow('YOLO Labeler', 1280, 720)         # 화면에 맞게 조절

        cv2.setMouseCallback('YOLO Labeler', self.mouse_callback)

        print("\n========== YOLO 라벨링 도구 ==========")
        print(f"이미지 폴더: {self.image_folder}")
        print(f"출력 폴더: {self.output_folder}")
        print(f"총 이미지: {len(self.image_files)}개")
        print("\n조작 방법:")
        print("  - 마우스로 드래그: 차량 박스 추가")
        print("  - 's' 키: 현재 이미지 저장 후 다음으로")
        print("  - 'a' 키: 이전 이미지 (저장 후 이동)")
        print("  - 'd' 키: 다음 이미지 (저장 없이 이동)")
        print("  - 'u' 키: 마지막 박스 삭제")
        print("  - 'c' 키: 모든 박스 삭제")
        print("  - 'q' 키: 종료")
        print("====================================\n")

        if not self.image_files:
            print("라벨링할 이미지가 없습니다.")
            return

        while True:
            self.display_image()
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("종료 중...")
                break

            elif key == ord('s'):
                # 저장하고 다음 이미지로
                self.save_annotations()
                self.current_idx += 1
                if self.current_idx >= len(self.image_files):
                    print("모든 이미지를 라벨링했습니다!")
                    break

            elif key == ord('a'):
                # 현재 것도 저장하고 이전 이미지로
                self.save_annotations()
                if self.current_idx > 0:
                    self.current_idx -= 1

            elif key == ord('d'):
                # 저장 없이 다음 이미지로
                self.current_idx += 1
                if self.current_idx >= len(self.image_files):
                    print("모든 이미지를 라벨링했습니다!")
                    break

            elif key == ord('u'):
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"삭제됨: {removed}")

            elif key == ord('c'):
                self.boxes = []
                print("모든 박스 삭제됨")

        cv2.destroyAllWindows()
        print("라벨링 완료!")


if __name__ == '__main__':
    # 프레임이 저장된 폴더
    image_folder = './frames'
    # 라벨이 저장될 폴더
    output_folder = './labels'

    labeler = YOLOLabeler(image_folder, output_folder)
    labeler.run()
