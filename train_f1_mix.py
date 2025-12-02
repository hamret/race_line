# from ultralytics import YOLO
#
# def main():
#     model = YOLO('yolo11l.pt')
#
#     results = model.train(
#         data=r'C:\Users\user\PycharmProjects\race_line\datasets\f1_mix.yaml',
#         epochs=50,
#         lr0=0.001,   # 기본 0.01 → 0.001로 낮춤
#         imgsz=960,
#         batch=8,
#         device=0,
#         workers=4,   # 문제 있으면 0으로 낮춰도 됨
#         patience=10,
#     )
#
# if __name__ == '__main__':
#     main()


from ultralytics import YOLO

def main():
    # COCO로 미리 학습된 YOLO11L
    model = YOLO('yolo11l.pt')

    results = model.train(
        data=r'C:\Users\user\PycharmProjects\race_line\datasets\my_f1.yaml',
        epochs=40,        # 25~40 사이에서 조절
        imgsz=960,        # 해상도 ↑ (학습도 960으로)
        batch=12,          # 4070이면 8~16 가능, 일단 8 안전
        device=0,
        workers=4,
        patience=10,
    )

if __name__ == '__main__':
    main()
