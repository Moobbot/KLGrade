# Pipeline: KL Detection (Detection-only)

Hướng dẫn end-to-end để tạo knee crops + nhãn KL (YOLO), tạo splits và huấn luyện detector (Ultralytics YOLO).

## Yêu cầu

- Đã cài đặt môi trường theo hướng dẫn ở README gốc (root).
- Thư mục dữ liệu:
  - Ảnh: `dataset/images/*.jpg|png`
  - Nhãn YOLO: `dataset/labels/*.txt` (dòng: `class cx cy w h`, normalized)
- Chạy lệnh từ thư mục gốc của repo: `E:\CaoHoc\thesis\KLGrade`

## 0. Kiểm tra dữ liệu đầu vào (khuyến nghị)

Sử dụng các tiện ích trong `check_dataset/` để rà soát dữ liệu trước khi tiền xử lý:

- `check_dataset/check_labels.py`: kiểm tra file nhãn YOLO hợp lệ (định dạng, số trường, giá trị trong [0,1] cho toạ độ chuẩn hoá nếu dùng, v.v.).
- `check_dataset/check_image_label.py`: đối chiếu ảnh và nhãn, phát hiện thiếu nhãn/ảnh.
- `check_dataset/visualize_yolo_boxes.py`: trực quan hoá bbox trên ảnh để phát hiện nhãn sai.
- `check_dataset/check_augment.py`: kiểm thử nhanh augment và vẽ kết quả để phát hiện lỗi pipeline augment.

Ví dụ:

```powershell
python check_dataset/check_image_label.py --img_dir dataset/dataset_v0/images --label_dir dataset/dataset_v0/labels
python check_dataset/visualize_yolo_boxes.py --img_dir dataset/dataset_v0/images --label_dir dataset/dataset_v0/labels --out_dir check_vis
```

## 1. Tiền xử lý: cắt vùng đầu gối, tái chiếu nhãn KL

Hai giai đoạn (đúng với dữ liệu của bạn):

1. Dùng nhãn đầu gối (labels-knee) để cắt ROI khớp gối từ ảnh toàn chân (crop vuông).
2. Tái chiếu nhãn KL (labels) vào knee crop để tạo nhãn YOLO cho detection.

Chạy script 2-stage (đầu vào mặc định: `dataset/dataset_v0/images`, `dataset/dataset_v0/labels-knee`, `dataset/dataset_v0/labels`):

```powershell
python pipeline/preprocess_knee_kl.py `
  --img_dir dataset/dataset_v0/images `
  --knee_labels dataset/dataset_v0/labels-knee `
  --kl_labels dataset/dataset_v0/labels `
  --knee_size 512 `
  --lesion_size 512
```

Kết quả:

- Knee crops: `processed/knee/images/*.jpg`
- Knee labels (KL tái chiếu, YOLO): `processed/knee/labels/*.txt`

Ghi chú hành vi (quan trọng): Knee ROI sau cùng sẽ được mở rộng/di chuyển để bao trọn mọi box KL nào giao nhau với knee đó (union → minimal square, clamp trong ảnh), có thêm đệm 2 px mỗi phía để box KL không dính viền.

Ghi chú chung:

- Số lớp lấy từ `config.py` (biến `CLASSES`).
- Kích thước resize dùng `IMG_SIZE` trong `config.py` (mặc định 512).

### Kiểm tra và lọc label sau tiền xử lý

Sau khi tạo knee crops và labels, chạy script kiểm tra để lọc label:

```powershell
python check_dataset/check_image_label.py --img_dir processed/knee/images --label_dir processed/knee/labels
```

Script này sẽ đối chiếu ảnh và nhãn, phát hiện và lọc các trường hợp thiếu nhãn/ảnh hoặc không khớp.

## 1.5. Tạo thống kê/soát xét nhanh (tuỳ chọn)

Dùng `pipeline/stats_labels.py` để thống kê phân phối lớp, số ảnh, số box, v.v.

```powershell
python pipeline/stats_labels.py --labels_dir processed/knee/labels
```

## 2. Tạo splits (bắt buộc trước khi train)

Sinh các file `splits/train.txt`, `splits/val.txt` (và `splits/test.txt` nếu cần) bằng script có sẵn:

```powershell
python check_dataset/split_dataset.py `
  --img_dir processed/knee/images `
  --label_dir processed/knee/labels `
  --out_dir splits `
  --train 0.7 `
  --val 0.15 `
  --test 0.15 `
  --seed 42
```

**Arguments:**

- `--img_dir`: thư mục chứa ảnh (default: `dataset/knee/images`)
- `--label_dir`: thư mục chứa labels YOLO (default: `dataset/knee/labels`)
- `--out_dir`: thư mục output cho split lists (default: `splits`)
- `--train`, `--val`, `--test`: tỷ lệ chia (mặc định: 0.7, 0.15, 0.15)
- `--seed`: random seed (mặc định: 42)

Kết quả: thư mục `splits/` chứa các danh sách tên file gốc (một tên mỗi dòng).

## 3. Huấn luyện detection (YOLO) trên knee crops

Khi một ảnh có nhiều box KL, hãy huấn luyện detector trên knee crops và nhãn KL đã tái chiếu:

Đầu vào:

- Ảnh: `processed/knee/images/*.jpg`
- Nhãn YOLO: `processed/knee/labels/*.txt`
- Splits: `splits/train.txt`, `splits/val.txt` (theo stem ảnh gốc)

Chạy train YOLO (Ultralytics):

```powershell
pip install ultralytics
python pipeline/train_det.py --img_dir processed/knee/images --splits_dir splits --model yolov8n.pt --epochs 100 --imgsz 512 --batch 16
```

Script sẽ tạo:

- `processed/det/train.txt`, `processed/det/val.txt` (danh sách ảnh tuyệt đối)
- `processed/det/dataset.yaml` (tên lớp từ `config.CLASSES`)

và gọi Ultralytics để train. Thư mục kết quả trong `processed/det/runs/`.

### (Tuỳ chọn) Preview đọc dữ liệu bằng `dataset.py` (có vẽ bbox)

Chạy script preview tách riêng (không train):

```powershell
python pipeline/preview_det_dataset.py `
  --img_dir processed/knee/images `
  --splits_dir splits `
  --split train `
  --n 5 `
  --out_dir check_vis/preview
```

Script chỉ duyệt thử N mẫu để phát hiện lỗi nhãn hoặc lỗi đường dẫn, không ảnh hưởng quá trình train.

## Mẹo & Gỡ lỗi

- Luôn kích hoạt môi trường ảo trước khi chạy:

```powershell
E:/CaoHoc/thesis/KLGrade/.venv/Scripts/Activate.ps1
```

- Nếu lỗi import `config`, hãy chắc đang chạy từ thư mục gốc repo.
- Thiếu thư viện: `pip install -r requirements.txt`.
