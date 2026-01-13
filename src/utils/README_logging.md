# Logging Utilities

Module tiện ích để quản lý log directories cho các training scripts.

## Chức năng

### 1. `get_next_log_dir(base_dir="log", module_name=None)`

Tự động tạo thư mục log có số thứ tự theo module, không ghi đè log cũ.

**Ví dụ:**
```python
from src.utils.logging_utils import get_next_log_dir

# Với module name
log_dir = get_next_log_dir("log", "kiocmil")
print(f"Logging to: {log_dir}")  # log/run_kiocmil_001

# Lần chạy tiếp theo
log_dir = get_next_log_dir("log", "kiocmil")  
# log/run_kiocmil_002

# Module khác
log_dir = get_next_log_dir("log", "detr")
# log/run_detr_001
```

**Format thư mục:**
- **Với module name:** `run_<tên_module>_<số>` 
  - Ví dụ: `run_kiocmil_001`, `run_kiocmil_002`, `run_detr_001`
- **Không có module name:** `run_<số>`
  - Ví dụ: `run_001`, `run_002`

### 2. `save_training_config(log_dir, args, additional_info=None)`

Lưu cấu hình training vào file JSON.

**Ví dụ:**
```python
from src.utils.logging_utils import save_training_config

config_path = save_training_config(
    log_dir, 
    args, 
    {"device": "cuda:0", "model": "resnet18"}
)
```

**File config.json:**
```json
{
  "timestamp": "2026-01-13 07:30:00",
  "args": {
    "epochs": 50,
    "batch_size": 8,
    ...
  },
  "device": "cuda:0",
  "model": "resnet18"
}
```

### 3. `setup_training_logging(base_dir, args, additional_info=None)`

Tạo log directory và lưu config trong một lần gọi.

**Ví dụ:**
```python
from src.utils.logging_utils import setup_training_logging

log_dir = setup_training_logging(
    "log", 
    args, 
    {"device": device, "model": "resnet18"}
)
```

## Sử dụng trong Training Scripts

### KIOCMIL Training

Script đã được cập nhật để tự động sử dụng versioned log directories với tên module:

```bash
python src/training/train_kiocmil.py --use_v2_dataset --epochs 50
```

Mỗi lần chạy sẽ tự động tạo:
- `log/run_kiocmil_001/` - lần chạy đầu
- `log/run_kiocmil_002/` - lần chạy thứ hai
- `log/run_kiocmil_003/` - lần chạy thứ ba
- ...

Mỗi thư mục chứa:
- `config.json` - cấu hình training
- Log files (nếu có)

### Training Scripts Khác

Để sử dụng trong script khác (ví dụ DETR):

```python
from src.utils.logging_utils import get_next_log_dir, save_training_config

# Trong __init__ của Trainer class
self.log_dir = get_next_log_dir("log", module_name="detr")
print(f"📁 Logging to: {self.log_dir}")  # log/run_detr_001

# Lưu config
save_training_config(
    self.log_dir,
    args,
    {"device": str(self.device), "model": self.model_name}
)
```

### Hoặc dùng setup_training_logging

```python
from src.utils.logging_utils import setup_training_logging

# Setup all-in-one
self.log_dir = setup_training_logging(
    "log",
    module_name="yolo",  # sẽ tạo run_yolo_001
    args=args,
    additional_info={"device": str(self.device)}
)
```

## Lợi ích

✅ **Không ghi đè log cũ** - Mỗi run có thư mục riêng  
✅ **Dễ so sánh** - Xem lại config của các lần chạy trước  
✅ **Tự động đánh số** - Không cần nhớ số thứ tự  
✅ **Reusable** - Dùng chung cho nhiều training scripts  
✅ **WandB friendly** - Log directory được lưu vào wandb config
