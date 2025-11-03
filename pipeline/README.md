# Pipeline: KL Classification

Hướng dẫn end-to-end để chuyển YOLO labels thành bộ dữ liệu phân loại (KL0–KL4), huấn luyện classifier và trực quan hóa Grad-CAM.

## Yêu cầu

- Đã cài đặt môi trường theo hướng dẫn ở README gốc (root).
- Thư mục dữ liệu:
  - Ảnh: `dataset/images/*.jpg|png`
  - Nhãn YOLO: `dataset/labels/*.txt` (dòng: `class cx cy w h`, normalized)
- Chạy lệnh từ thư mục gốc của repo: `E:\CaoHoc\thesis\KLGrade`

## 1. Tiền xử lý: cắt vùng đầu gối thành ảnh phân loại

Bạn có 2 cách tiền xử lý, tuỳ bộ nhãn của bạn:

- Hai giai đoạn (đúng với dữ liệu của bạn):
  1) Dùng nhãn đầu gối (labels-knee) để cắt ROI khớp gối từ ảnh toàn chân.
  2) Dùng nhãn KL (labels) để cắt patch tổn thương cho phân loại.

  Chạy script 2-stage mới (đầu vào mặc định: `dataset/dataset_v0/images`, `dataset/dataset_v0/labels-knee`, `dataset/dataset_v0/labels`):

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
  - Lesion crops: `processed/classification/images/*.jpg`
  - CSV: `processed/classification/labels.csv`

  Ghi chú hành vi (quan trọng):
  - Mỗi knee ROI ban đầu được tạo thành crop vuông (square) tâm theo box knee, sau đó SẼ ĐƯỢC MỞ RỘNG/DI CHUYỂN để BAO TRỌN các box KL bất kỳ nào GIAO NHAU với knee đó (union → minimal square, clamp trong biên ảnh). Nhờ vậy, các box KL giao nhau sẽ nằm hoàn toàn bên trong knee crop sau cùng và không bị cắt mất.

- Một giai đoạn (nếu bạn chỉ có nhãn KL trên ảnh gốc):
  Sinh crop (mặc định 512x512, có thể đổi bằng `--size`) và CSV nhãn phân loại trực tiếp từ nhãn KL:

```powershell
python pipeline/preprocess_crops.py
# hoặc tuỳ chỉnh kích thước crop:
python pipeline/preprocess_crops.py --size 512
```

Kết quả:

- Ảnh crop: `processed/classification/images/*.jpg`
- CSV: `processed/classification/labels.csv` (cột: path,label; label là id lớp 0..4)

Ghi chú chung:

- Số lớp lấy từ `config.py` (biến `CLASSES`).
- Kích thước resize dùng `IMG_SIZE` trong `config.py` (mặc định 512).

## 2. (Tuỳ chọn) Tích hợp ảnh sinh (generative)

Nếu bạn có ảnh tổng hợp (GAN/Diffusion), đặt theo lớp:

```text
processed/generative/
  KL0/*.jpg
  KL1/*.jpg
  KL2/*.jpg
  KL3/*.jpg
  KL4/*.jpg
```

Tạo CSV mới bao gồm cả dữ liệu sinh:

```powershell
python pipeline/integrate_generated.py `
  --gen_dir processed/generative `
  --csv processed/classification/labels.csv `
  --out_csv processed/classification/labels_with_generated.csv
```

Lưu ý khi dùng splits: khi huấn luyện với `splits/train.txt` và `splits/val.txt`, chỉ các hàng trong CSV có tên file (stem) khớp với các stem trong file split mới được đưa vào train/val. Nếu ảnh sinh không mang stem gốc (ví dụ không theo dạng `<stem>_obj...jpg`), bạn cần:

- Đặt tên file sinh để stem trùng với stem ảnh gốc thuộc split tương ứng, hoặc
- Thêm stem của ảnh sinh vào `splits/train.txt` hoặc `splits/val.txt` theo ý định.

## Tạo splits (bắt buộc trước khi train)

Sinh các file `splits/train.txt`, `splits/val.txt` (và `splits/test.txt` nếu cần) bằng script có sẵn:

```powershell
python check_dataset/split_dataset.py
```

Kết quả: thư mục `splits/` chứa các danh sách tên file gốc (một tên mỗi dòng).

## 3. Huấn luyện classifier

Mặc định sẽ dùng `processed/classification/labels.csv`. Nếu đã tạo CSV có dữ liệu sinh, thêm `--csv`.

Ví dụ huấn luyện ResNet50, 30 epoch, lưu mô hình:

```powershell
python pipeline/train_cls.py `
  --backbone resnet50 `
  --epochs 30 `
  --batch 16 `
  --size 512 `
  --sampler `
  --class_weights `
  --splits_dir splits `
  --save models/cls_resnet50.pt
```

Tùy chọn quan trọng:

- `--backbone`: `resnet50` | `resnet101` | `efficientnet_b0`
- `--sampler`: bật WeightedRandomSampler (xử lý mất cân bằng lớp)
- `--class_weights`: bật trọng số lớp cho CrossEntropyLoss
- `--mixup 0.2` hoặc `--cutmix 0.2`: bật augmentations nâng cao
- `--csv <path>`: chọn CSV khác (ví dụ với dữ liệu sinh)

 Ghi chú: `train_cls.py` yêu cầu có sẵn `splits/train.txt` và `splits/val.txt`. Nếu chưa có, hãy tạo bằng `check_dataset/split_dataset.py`.

Hoặc chỉ định trực tiếp đường dẫn file split:

```powershell
python pipeline/train_cls.py `
  --backbone cnn_basic `
  --csv processed/classification/labels.csv `
  --train_split splits/train.txt `
  --val_split_file splits/val.txt `
  --size 512 `
  --save models/cls_cnn_basic.pt
```

Lưu ý cấu hình 512x512:

- Bạn KHÔNG cần tạo lại crop nếu chỉ thay đổi kích thước train; script sẽ resize on-the-fly qua augmentations.
- 512x512 tốn VRAM hơn; giảm `--batch` nếu gặp OOM.

## (Tuỳ chọn) Resize-only (không crop lại)

Nếu dữ liệu của bạn đã là ảnh vùng gối (đã crop), nhưng bạn muốn chuẩn hoá kích thước file trên đĩa về 512x512 mà không crop lại, dùng script sau:

```powershell
# Resize vuông 512x512 (bị kéo giãn nếu ảnh không vuông)
python pipeline/resize_images.py `
  --in_dir processed/classification/images `
  --out_dir processed/classification/images_512 `
  --size 512

# Hoặc giữ tỉ lệ, pad viền (letterbox) về 512x512
python pipeline/resize_images.py `
  --in_dir processed/classification/images `
  --out_dir processed/classification/images_512_lb `
  --size 512 `
  --keep_aspect
```

 

## 4. Grad-CAM: trực quan hoá vùng quan trọng

Sinh heatmap Grad-CAM chồng lên ảnh crop để giải thích dự đoán.

Ví dụ với ResNet50 (layer gợi ý: `layer4.2`):

```powershell
python pipeline/gradcam.py `
  --model models/cls_resnet50.pt `
  --image processed/classification/images/<ten_anh>.jpg `
  --backbone resnet50 `
  --layer layer4.2 `
  --size 512 `
  --out gradcam_<ten_anh>.jpg
```

Lưu ý:

- Với EfficientNet, cần chỉ định tên layer phù hợp trong backbone (ví dụ một block trong `features`). Nếu chỉ định sai layer, script sẽ báo lỗi không tìm thấy layer.

## Mẹo & Gỡ lỗi

- Luôn kích hoạt môi trường ảo trước khi chạy:

```powershell
E:/CaoHoc/thesis/KLGrade/.venv/Scripts/Activate.ps1
```

- Nếu lỗi import `config`, hãy chắc đang chạy từ thư mục gốc repo.
- Thiếu thư viện: `pip install -r requirements.txt`.
