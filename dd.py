import os, shutil, random
from pathlib import Path

base = r'C:\Users\user\PycharmProjects\race_line'

frames_dir = Path(base) / 'frames'   # 네가 라벨링한 이미지들
labels_dir = Path(base) / 'labels'   # YOLO txt 라벨들

out_images_train = Path(base) / 'my_f1/images/train'
out_images_val   = Path(base) / 'my_f1/images/val'
out_labels_train = Path(base) / 'my_f1/labels/train'
out_labels_val   = Path(base) / 'my_f1/labels/val'

for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
    p.mkdir(parents=True, exist_ok=True)

# 이미지 목록 (이미 라벨링된 140장 기준)
images = sorted([f for f in frames_dir.iterdir()
                 if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

print('총 이미지 수:', len(images))

random.shuffle(images)

# 140장 기준 80%/20% → 112 / 28 정도
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

def copy_pair(img_paths, img_out_dir, label_out_dir):
    for img_path in img_paths:
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.copy2(img_path, img_out_dir / img_path.name)
            shutil.copy2(label_path, label_out_dir / label_path.name)
        else:
            print('라벨 없음, 스킵:', img_path)

copy_pair(train_images, out_images_train, out_labels_train)
copy_pair(val_images, out_images_val, out_labels_val)

print('train 이미지 수:', len(list(out_images_train.glob('*'))))
print('val 이미지 수:', len(list(out_images_val.glob('*'))))
print('Done.')
