# Tiền Xử Lý Labels: Phân Tách Class thành Gai Xương và Khe Khớp

## Tổng Quan

Trong dataset X-quang đầu gối, mỗi ảnh có thể chứa nhiều bounding box cùng một class KL (Kellgren-Lawrence), nhưng thực tế chúng đại diện cho các cấu trúc khác nhau:

- **Gai xương (Osteophyte)**: Các mỏm xương nhỏ, gần vuông
- **Khe khớp (Joint Space)**: Khe hẹp giữa các xương, dài và mỏng theo chiều ngang

Script `class_split_report.py` tự động phân tách các box cùng class thành 2 nhánh con (tag "a" và "b") dựa trên đặc điểm hình học, giúp:

- Tăng độ chi tiết trong annotation
- Hỗ trợ training model phân biệt các cấu trúc khác nhau
- Dễ dàng kiểm tra và validate dữ liệu

## Quy Trình Xử Lý

### 1. Đọc Labels Gốc

Input: Thư mục chứa file label YOLO format chuẩn (`processed/knee/labels/`)

Format mỗi dòng: `class_id x_center y_center width height`

- Tất cả giá trị normalized (0.0 - 1.0)
- Ví dụ: `3 0.088764 0.733708 0.105618 0.134831`

### 2. Phân Loại Box (Bước 1: Xét Riêng Từng Box)

Với mỗi box, script tính toán:

- **Tỷ lệ w/h**: `ratio = width / height`
- **Diện tích**: `area = width * height`

Sau đó áp dụng quy tắc phân loại **chỉ khi rõ ràng**:

#### Quy Tắc Rõ Ràng

**Tag "a" (Gai xương)** - khi thỏa một trong:

- `ratio < 1.2` (gần vuông)
- `area < 0.01` (nhỏ)

**Tag "b" (Khe khớp)** - khi thỏa một trong:

- `ratio > 2.0` (dài, mỏng)
- `area > 0.03` (lớn)

**Không rõ ràng** - nếu không thỏa điều kiện trên → `suffix = None` (sẽ xử lý ở bước 2)

### 3. Xử Lý Box Không Rõ Ràng (Bước 2: So Sánh Tương Đối)

Khi có **2 box cùng class** và một hoặc cả hai không rõ ràng, script áp dụng rule so sánh:

#### Quy Tắc So Sánh

- **Box có width lớn hơn VÀ không nằm ở viền ảnh** → Tag "b" (Khe khớp)
- **Box còn lại** → Tag "a" (Gai xương)

**Kiểm tra viền ảnh**: Box được coi là ở viền nếu khoảng cách từ viền < 1% (threshold = 0.01)

**Fallback**: Nếu vẫn không xác định được → dựa trên ratio: `ratio < 1.5 → "a"`, else → "b"

### 4. Chuyển Đổi Sang Format YOLO Chuẩn

Sau khi phân loại thành tag "a"/"b", script tự động chuyển đổi sang class_id YOLO chuẩn (0-9) theo mapping trong `config.CLASSES_LABEL_NEW`:

- `"0a"` → class_id `0`, `"0b"` → class_id `1`
- `"1a"` → class_id `2`, `"1b"` → class_id `3`
- `"2a"` → class_id `4`, `"2b"` → class_id `5`
- `"3a"` → class_id `6`, `"3b"` → class_id `7`
- `"4a"` → class_id `8`, `"4b"` → class_id `9`

### 5. Tạo Label Mới

Output: Thư mục chứa file label mới (`processed/knee/labels_new/`)

Format mỗi dòng: `class_id x_center y_center width height` (YOLO chuẩn với class_id số nguyên)

Ví dụ:

```txt
6 0.088764 0.733708 0.105618 0.134831
7 0.284831 0.633708 0.338202 0.089888
```

(Trong đó: `6` = "3a", `7` = "3b" theo mapping)

## Quy Tắc Phân Loại Chi Tiết

### Tag "a" - Gai Xương (Osteophyte)

- **Đặc điểm**: Nhỏ, gần vuông hoặc dọc
- **Điều kiện rõ ràng**:
  - `w/h < 1.2` (gần vuông)
  - HOẶC `area < 0.01` (nhỏ)
- **Điều kiện so sánh** (khi không rõ ràng):
  - Box có width nhỏ hơn trong cặp 2 box cùng class
  - HOẶC box có width lớn hơn nhưng nằm ở viền ảnh

### Tag "b" - Khe Khớp (Joint Space)

- **Đặc điểm**: Rộng theo chiều ngang, lớn hơn gai xương
- **Điều kiện rõ ràng**:
  - `w/h > 2.0` (dài, mỏng)
  - HOẶC `area > 0.03` (lớn)
- **Điều kiện so sánh** (khi không rõ ràng):
  - Box có width lớn hơn VÀ không nằm ở viền ảnh

## Cách Sử Dụng

### 1. Chạy Script Phân Tách

```powershell
# Sử dụng tham số mặc định
python check_dataset/class_split_report.py

# Chỉ xử lý class 2 và 3
python check_dataset/class_split_report.py --classes 2 3

# Chỉ định thư mục input/output
python check_dataset/class_split_report.py `
  --labels-dir processed/knee/labels `
  --save-dir processed/knee/labels_new
```

### 2. Kiểm Tra Kết Quả

Script sẽ in ra:

- Thống kê số lượng box theo từng class
- Ví dụ các box đã được phân loại (với ratio và area)
- **Cảnh báo cho box không rõ ràng**: Hiển thị file, class, ratio, area và suffix được gán (fallback)

### 3. Visualize Labels Mới

Sử dụng `visualize_yolo_boxes.py` để xem kết quả trên ảnh:

```powershell
python check_dataset/visualize_yolo_boxes.py `
  --img_dir processed/knee/images `
  --label_dir processed/knee/labels_new `
  --out_dir check_vis/label_new
```

Script này hỗ trợ cả format chuẩn và format mở rộng (có suffix), nên có thể visualize trực tiếp label mới.

## Ví Dụ

### Input (Label Gốc)

File: `1.2.392.200036.9107.307.24972.20220914.81908.1033568_0.txt`

```txt
3 0.088764 0.733708 0.105618 0.134831
3 0.284831 0.633708 0.338202 0.089888
```

### Phân Tích

- **Box 1**: `w=0.105618, h=0.134831`

  - Ratio: `0.105618 / 0.134831 ≈ 0.78` (< 1.2)
  - Area: `0.105618 * 0.134831 ≈ 0.014` (< 0.03)
  - → Tag "a" (Gai xương)

- **Box 2**: `w=0.338202, h=0.089888`
  - Ratio: `0.338202 / 0.089888 ≈ 3.77` (> 2.0)
  - Area: `0.338202 * 0.089888 ≈ 0.030` (≈ 0.03)
  - → Tag "b" (Khe khớp)

### Output (Label Mới)

File: `1.2.392.200036.9107.307.24972.20220914.81908.1033568_0.txt`

```txt
6 0.088764 0.733708 0.105618 0.134831
7 0.284831 0.633708 0.338202 0.089888
```

**Giải thích:**

- `6` = "3a" (KL3-a, gai xương) theo mapping `CLASSES_LABEL_NEW`
- `7` = "3b" (KL3-b, khe khớp) theo mapping `CLASSES_LABEL_NEW`

## Cấu Hình

### Tham Số Mặc Định

- `--labels-dir`: `processed/knee/labels`
- `--save-dir`: `processed/knee/labels_new`
- `--max-split`: `2` (chỉ hỗ trợ a/b)
- `--limit`: `10` (số ví dụ hiển thị)

### Mapping Class (config.py)

Script tự động chuyển đổi tag sang class_id YOLO theo `CLASSES_LABEL_NEW` trong `config.py`:

```python
CLASSES_LABEL_NEW = {
    0: "KL0-a",  # class_id 0 tương ứng với "0a"
    1: "KL0-b",  # class_id 1 tương ứng với "0b"
    2: "KL1-a",  # class_id 2 tương ứng với "1a"
    3: "KL1-b",  # class_id 3 tương ứng với "1b"
    4: "KL2-a",  # class_id 4 tương ứng với "2a"
    5: "KL2-b",  # class_id 5 tương ứng với "2b"
    6: "KL3-a",  # class_id 6 tương ứng với "3a" (gai xương)
    7: "KL3-b",  # class_id 7 tương ứng với "3b" (khe khớp)
    8: "KL4-a",  # class_id 8 tương ứng với "4a"
    9: "KL4-b",  # class_id 9 tương ứng với "4b"
}
```

**Công thức chuyển đổi:**

- `class_id = base_class * 2 + (0 nếu "a", 1 nếu "b")`
- Ví dụ: `"3a" → 3 * 2 + 0 = 6`, `"3b" → 3 * 2 + 1 = 7`

## Lưu Ý

1. **Phân Loại 2 Bước**:

   - Bước 1: Xét riêng từng box, chỉ phân loại khi rõ ràng
   - Bước 2: So sánh box không rõ ràng với box khác cùng class (nếu có)

2. **Đặc Trưng Khe Khớp**: Khe khớp (b) rộng theo chiều ngang và lớn hơn gai xương (a), nên logic ưu tiên kiểm tra width và vị trí viền ảnh.

3. **Cảnh Báo**: Script chỉ cảnh báo cho box không rõ ràng (phải dùng fallback).

4. **Xử Lý Nhiều Box**: Ảnh có thể có nhiều box cùng class, script xử lý tất cả. Khi có 2 box cùng class, sẽ áp dụng rule so sánh cho box không rõ ràng.

5. **Format Hỗ Trợ**: Script visualize (`visualize_yolo_boxes.py`) tự động nhận diện cả format chuẩn và format mở rộng, không cần chỉnh sửa.

6. **Backward Compatibility**: Label mới vẫn giữ nguyên format YOLO (chỉ thêm suffix vào class_id), nên có thể sử dụng với các tool YOLO khác (cần normalize lại class_id nếu cần).

## Tài Liệu Liên Quan

- `check_dataset/class_split_report.py`: Script phân tách class
- `check_dataset/visualize_yolo_boxes.py`: Script visualize labels
- `config.py`: Cấu hình class mapping
