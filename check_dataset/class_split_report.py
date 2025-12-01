"""
Phân tách class YOLO labels thành nhánh con (a/b) dựa trên đặc điểm hình học.

Quy tắc:
- "a" = gai xương (w/h < 1.2 hoặc area < 0.01)
- "b" = khe khớp (w/h > 2.0 hoặc area > 0.03)
- 2 box cùng class khác loại: box lớn → "b", box nhỏ → "a"
- 2 box cùng loại: giữ nguyên

Output: "{class}{suffix} x y w h"
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gán nhãn a/b: a=gai xương, b=khe khớp"
    )
    parser.add_argument("--labels-dir", type=Path, required=None, default="processed/knee/labels",
                       help="Thư mục chứa file label YOLO gốc (*.txt)")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                       help="Chỉ xử lý các class cụ thể (ví dụ: --classes 2 3)")
    parser.add_argument("--limit", type=int, default=10,
                       help="Số lượng ví dụ hiển thị cho mỗi class trong báo cáo")
    parser.add_argument("--max-split", type=int, default=2,
                       help="Số nhánh tối đa (mặc định 2: a/b)")
    parser.add_argument("--save-dir", type=Path, default="processed/knee/labels_new",
                       help="Thư mục lưu file label mới với tag a/b")
    return parser.parse_args()


def iter_label_files(labels_dir: Path) -> Iterable[Path]:
    """Lặp qua file .txt trong thư mục (đã sắp xếp)."""
    for path in sorted(labels_dir.glob("*.txt")):
        if path.is_file():
            yield path


def load_boxes(label_path: Path) -> List[Tuple[int, List[float]]]:
    """Đọc file label YOLO: class_id x y w h (normalized)."""
    boxes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:5]))
            boxes.append((cls, coords))
    return boxes


def classify_box(w: float, h: float, area: float) -> str:
    """
    Phân loại box: "a" (gai xương) hoặc "b" (khe khớp).
    
    Chỉ phân loại khi rõ ràng:
    - "a": w/h < 1.2 hoặc area < 0.01
    - "b": w/h > 2.0 hoặc area > 0.03
    
    Nếu không rõ ràng → trả về None để xử lý sau.
    """
    if h <= 0:
        return "a"

    ratio = w / h
    
    # Rõ ràng là "a" (gai xương)
    if ratio < 1.2 or area < 0.01:
        return "a"
    
    # Rõ ràng là "b" (khe khớp)
    if ratio > 2.0 or area > 0.03:
        return "b"
    
    # Không rõ ràng → trả về None
    return None


def is_at_edge(x: float, y: float, w: float, h: float, threshold: float = 0.01) -> bool:
    """
    Kiểm tra box có nằm ở viền ảnh không.
    
    Args:
        x, y: Tọa độ center (normalized)
        w, h: Width, height (normalized)
        threshold: Ngưỡng khoảng cách từ viền (mặc định 0.05 = 5%)
    
    Returns:
        True nếu box gần viền ảnh
    """
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    
    # Kiểm tra gần viền trái, phải, trên, dưới
    return (x_min < threshold or x_max > 1.0 - threshold or
            y_min < threshold or y_max > 1.0 - threshold)


def assign_relative_group(boxes: List[dict]):
    """
    Xử lý box không rõ ràng: so sánh width và kiểm tra vị trí viền.
    
    Logic:
    - Nếu box có width lớn hơn VÀ không nằm ở viền ảnh → "b" (khe khớp)
    - Box còn lại → "a" (gai xương)
    """
    if len(boxes) != 2:
        return

    b1, b2 = boxes
    
    # Chỉ xử lý box có suffix = None (không rõ ràng)
    if b1["suffix"] is not None and b2["suffix"] is not None:
        return  # Cả 2 đều đã rõ ràng
    
    # Lấy thông tin box
    x1, y1, w1, h1 = b1["coords"]
    x2, y2, w2, h2 = b2["coords"]
    
    # Kiểm tra box nào có width lớn hơn và không ở viền
    at_edge1 = is_at_edge(x1, y1, w1, h1)
    at_edge2 = is_at_edge(x2, y2, w2, h2)
    
    # Box có width lớn hơn và không ở viền → "b"
    if w1 > w2 and not at_edge1:
        b1["suffix"] = "b"
        if b2["suffix"] is None:
            b2["suffix"] = "a"
    elif w2 > w1 and not at_edge2:
        b2["suffix"] = "b"
        if b1["suffix"] is None:
            b1["suffix"] = "a"
    else:
        # Nếu không thỏa điều kiện trên, xét width đơn giản
        if w1 > w2:
            if b1["suffix"] is None:
                b1["suffix"] = "b"
            if b2["suffix"] is None:
                b2["suffix"] = "a"
        else:
            if b1["suffix"] is None:
                b1["suffix"] = "a"
            if b2["suffix"] is None:
                b2["suffix"] = "b"


def build_reports(labels_dir, allowed_classes, max_split):
    """
    Xử lý labels và tạo output.
    
    Returns:
        (class_entries, warnings, exported_lines)
    """
    class_entries = defaultdict(list)
    warnings = []
    exported_lines = defaultdict(list)

    for label_file in iter_label_files(labels_dir):
        boxes_loaded = load_boxes(label_file)

        class_map = defaultdict(list)
        for cls, coords in boxes_loaded:
            if allowed_classes and cls not in allowed_classes:
                continue
            class_map[cls].append(coords)

        for cls, coords_list in class_map.items():
            entries = []
            for coords in coords_list:
                x, y, w, h = coords
                area = w * h
                suffix = classify_box(w, h, area)

                entries.append(
                    {
                        "cls": cls,
                        "coords": coords,
                        "suffix": suffix,
                        "area": area,
                        "ratio": w / h if h > 0 else 0,
                        "file": label_file.name,
                    }
                )

            # Xử lý box không rõ ràng (suffix = None)
            if len(entries) == 2:
                assign_relative_group(entries)
            
            # Đảm bảo tất cả box đều có suffix (fallback nếu vẫn None)
            for ent in entries:
                if ent["suffix"] is None:
                    # Fallback: dựa trên ratio
                    ent["suffix"] = "a" if ent["ratio"] < 1.5 else "b"
                    # Cảnh báo box không rõ ràng
                    warnings.append(
                        f"{label_file.name}: class {cls} box không rõ ràng "
                        f"(ratio={ent['ratio']:.2f}, area={ent['area']:.3f}) "
                        f"-> fallback {ent['suffix']}"
                    )
                
                tag = f"{ent['cls']}{ent['suffix']}"
                class_entries[cls].append(ent)
                out_line = f"{tag} " + " ".join(f"{v:.6f}" for v in ent["coords"])
                exported_lines[label_file.name].append(out_line)

    return class_entries, warnings, exported_lines


def print_report(class_entries, warnings, limit):
    """In báo cáo thống kê và cảnh báo."""
    for cls in sorted(class_entries):
        print(f"\nClass {cls}")
        for e in class_entries[cls][:limit]:
            print(
                f"  {e['file']} -> {e['cls']}{e['suffix']} "
                f"(ratio={e['ratio']:.2f}, area={e['area']:.3f})  "
                f"{e['coords']}"
            )

    if warnings:
        print("\n=== CANH BAO ===")
        for w in warnings:
            print("  *", w)


def main():
    args = parse_args()

    class_entries, warnings, exported = build_reports(
        args.labels_dir, args.classes, args.max_split
    )

    print_report(class_entries, warnings, args.limit)

    if args.save_dir:
        args.save_dir.mkdir(exist_ok=True, parents=True)
        written = 0
        for fname, lines in exported.items():
            if lines:
                with (args.save_dir / fname).open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                written += 1
        print(f"\nDa luu {written} file moi vao {args.save_dir}")


if __name__ == "__main__":
    main()