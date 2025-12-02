from ultralytics import YOLO

def main():
    model = YOLO('yolo11l.pt')

    results = model.train(
        data=r'C:\Users\user\PycharmProjects\race_line\datasets\f1_mix.yaml',
        epochs=30,
        imgsz=960,
        batch=16,
        device=0,
        workers=4,   # 문제 있으면 0으로 낮춰도 됨
        patience=10,
    )

if __name__ == '__main__':
    main()
