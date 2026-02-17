"""
Update original knee labels based on expansion log.
Reads processed/knee/expanded_files.json and updates the original label files
in dataset/dataset_v0/labels-knee/ with the expanded coordinates.
"""

import sys
from pathlib import Path
import json
import cv2
import argparse
from tqdm import tqdm
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel coordinates to YOLO format."""
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    n_cx = cx / img_w
    n_cy = cy / img_h
    n_w = w / img_w
    n_h = h / img_h

    # Clamp to [0,1]
    n_cx = np.clip(n_cx, 0, 1)
    n_cy = np.clip(n_cy, 0, 1)
    n_w = np.clip(n_w, 0, 1)
    n_h = np.clip(n_h, 0, 1)

    return n_cx, n_cy, n_w, n_h


def load_yolo_boxes(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    boxes.append(
                        {
                            "class_id": int(parts[0]),
                            "x": float(parts[1]),
                            "y": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        }
                    )
                except ValueError:
                    continue
    return boxes


def save_yolo_labels(boxes, output_path: Path):
    if not boxes:
        output_path.write_text("")
        return

    lines = []
    for box in boxes:
        line = f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}"
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n")


def update_labels(expanded_log: Path, dataset_dir: Path, dry_run: bool = False):
    print(f"Log: {expanded_log}")
    print(f"Dataset: {dataset_dir}")
    print(f"Dry run: {dry_run}")

    with open(expanded_log, "r") as f:
        expanded_files = json.load(f)

    print(f"Found {len(expanded_files)} entries to update")

    # Group by file stem to minimize file I/O
    updates_by_stem = {}
    for entry in expanded_files:
        # entry['file'] is like "stem_knee0"
        stem = entry["file"].rsplit("_knee", 1)[0]
        knee_idx = int(entry["file"].split("_knee")[-1])

        if stem not in updates_by_stem:
            updates_by_stem[stem] = []
        updates_by_stem[stem].append(
            {
                "knee_idx": knee_idx,
                "expanded_box": entry["expanded_box"],  # [x1, y1, x2, y2]
            }
        )

    updated_count = 0

    for stem, updates in tqdm(updates_by_stem.items()):
        # Find image for dimensions
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            p = dataset_dir / "images" / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break

        if not img_path:
            print(f"Warning: Image not found for {stem}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        h, w = img.shape[:2]

        # Load existing labels
        label_path = dataset_dir / "labels-knee" / f"{stem}.txt"
        boxes = load_yolo_boxes(label_path)

        if not boxes:
            print(f"Warning: No labels found for {stem}")
            continue

        # Apply updates
        modified = False
        for update in updates:
            idx = update["knee_idx"]
            if idx >= len(boxes):
                print(
                    f"Warning: Knee index {idx} out of range for {stem} (len={len(boxes)})"
                )
                continue

            ex_box = update["expanded_box"]
            nx, ny, nw, nh = pixel_to_yolo(
                ex_box[0], ex_box[1], ex_box[2], ex_box[3], w, h
            )

            # Update only if significantly different?
            # Or just update unconditionally as requested
            boxes[idx]["x"] = nx
            boxes[idx]["y"] = ny
            boxes[idx]["w"] = nw
            boxes[idx]["h"] = nh
            modified = True

        if modified and not dry_run:
            save_yolo_labels(boxes, label_path)
            updated_count += 1

    print(f"\nUpdated {updated_count} label files.")


def main():
    parser = argparse.ArgumentParser(
        description="Update original knee labels from expansion log"
    )
    parser.add_argument(
        "--expanded_log", type=str, required=True, help="Path to expanded_files.json"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to dataset_v0"
    )
    parser.add_argument("--dry_run", action="store_true", help="Don't write changes")

    args = parser.parse_args()

    update_labels(Path(args.expanded_log), Path(args.dataset_dir), args.dry_run)


if __name__ == "__main__":
    main()
