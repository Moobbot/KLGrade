import os
import cv2
import albumentations as A
from tqdm import tqdm
from dataset import YoloDataset
from albumentations.pytorch.transforms import ToTensorV2


def check_augmentation(dataset, num_trials=3):
    """
    Kiểm tra toàn bộ dataset qua augment xem có lỗi gì không.
    - num_trials: số lần thử augment mỗi ảnh (để random nhiều biến thể)
    """
    total_images = len(dataset)
    total_boxes = 0
    total_aug_boxes = 0
    errors = []
    empty_after_aug = 0

    print(f"🔍 Checking {total_images} images × {num_trials} augmentations...")

    for i in tqdm(range(total_images)):
        img_name = dataset.image_files[i]
        try:
            image = cv2.imread(os.path.join(dataset.img_dir, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            # đọc bbox gốc
            label_path = os.path.join(
                dataset.label_dir, os.path.splitext(img_name)[0] + ".txt"
            )
            boxes = []
            labels = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        c, x, y, bw, bh = map(float, line.strip().split())
                        x1 = (x - bw / 2) * w
                        y1 = (y - bh / 2) * h
                        x2 = (x + bw / 2) * w
                        y2 = (y + bh / 2) * h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(c))
            total_boxes += len(boxes)

            # thử augment nhiều lần
            for _ in range(num_trials):
                try:
                    transformed = dataset.transform(
                        image=image, bboxes=boxes, class_labels=labels
                    )
                    aug_boxes = transformed["bboxes"]
                    total_aug_boxes += len(aug_boxes)

                    # kiểm tra box còn nằm trong ảnh
                    for x1, y1, x2, y2 in aug_boxes:
                        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                            errors.append(
                                f"{img_name}: out-of-bound box ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                            )

                    # kiểm tra mất box
                    if len(aug_boxes) == 0 and len(boxes) > 0:
                        empty_after_aug += 1

                except Exception as e:
                    errors.append(f"{img_name}: augmentation error → {str(e)}")

        except Exception as e:
            errors.append(f"{img_name}: load error → {str(e)}")

    # thống kê
    print("\n===== SUMMARY =====")
    print(f"Total images: {total_images}")
    print(f"Total original boxes: {total_boxes}")
    print(f"Total aug boxes (all trials): {total_aug_boxes}")
    print(f"Images with empty boxes after aug: {empty_after_aug}")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\nSome examples of errors:")
        for e in errors[:10]:
            print(" -", e)
        with open("augmentation_errors.log", "w", encoding="utf-8") as f:
            f.write("\n".join(errors))
        print("\n⚠️ Full log saved to: augmentation_errors.log")
    else:
        print("✅ No errors detected. All augmentations are valid!")


if __name__ == "__main__":
    image_size = 512
    # Định nghĩa augment giống khi train
    # Không dùng - không phù hợp ảnh y tế A.Rotate(limit=20, p=1),

    transform = transform = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.15, p=0.5
            ),
            A.GaussNoise(var_limit=(5, 15), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.01,
            clip=True,  # auto clamp về biên ảnh
            check_each_transform=False,  # bỏ kiểm tra nghiêm ngặt mỗi bước
        ),
    )
    dataset = YoloDataset(
        img_dir="dataset/images", label_dir="dataset/labels", transform=transform
    )

    check_augmentation(dataset, num_trials=3)
