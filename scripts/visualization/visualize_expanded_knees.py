"""
Visualize expanded knee regions.
Draws:
- Original Knee Box (Red)
- Expanded Knee Box (Green)
- KL Labels (Blue) pointing to the center
"""

import sys
from pathlib import Path
import cv2
import json
import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize expanded knee regions",
        epilog="Draws Original Box (Red), Expanded Box (Green), and KL Labels (Blue)",
    )

    parser.add_argument(
        "--expanded-log",
        type=str,
        default="processed/knee/expanded_files.json",
        help="Path to expanded_files.json (default: processed/knee/expanded_files.json)",
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/dataset_v0",
        help="Dataset directory (default: datasets/dataset_v0)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed/knee/expanded_visualization",
        help="Output directory (default: processed/knee/expanded_visualization)",
    )

    args = parser.parse_args()

    root_dir = Path(".")
    expanded_log_path = root_dir / args.expanded_log
    dataset_dir = root_dir / args.dataset_dir
    output_dir = root_dir / args.output_dir

    if not expanded_log_path.exists():
        print(f"Error: {expanded_log_path} not found")
        return

    with open(expanded_log_path, "r") as f:
        expanded_files = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(expanded_files)} expanded files...")

    for entry in tqdm(expanded_files):
        file_id = entry["file"]  # e.g. stem_knee0
        # Recover info
        stem = file_id.rsplit("_knee", 1)[0]
        knee_idx = int(file_id.split("_knee")[-1])

        # Find image
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            p = dataset_dir / "images" / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break

        if not img_path:
            print(f"Image not found for {stem}")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Draw Original Box (Red)
        obox = entry["original_box"]
        ox1, oy1, ox2, oy2 = map(int, obox)
        cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
        cv2.putText(
            img,
            "Original",
            (ox1, oy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        # Draw Expanded Box (Green)
        ebox = entry["expanded_box"]
        ex1, ey1, ex2, ey2 = map(int, ebox)
        cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (0, 255, 0), 3)
        cv2.putText(
            img,
            "Expanded",
            (ex1, ey2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Draw KL Labels (Blue)
        label_path = dataset_dir / "labels" / f"{stem}.txt"
        kl_boxes = load_yolo_boxes(label_path)

        for kl in kl_boxes:
            kx = int(kl["x"] * w)
            ky = int(kl["y"] * h)
            kw = int(kl["w"] * w)
            kh = int(kl["h"] * h)

            # Check if this KL label is inside the Expanded Box
            # Drawing all KL labels might be cluttered, but useful to see context

            cv2.circle(img, (kx, ky), 5, (255, 0, 0), -1)
            # Draw box
            kx1 = int(kx - kw / 2)
            ky1 = int(ky - kh / 2)
            kx2 = int(kx + kw / 2)
            ky2 = int(ky + kh / 2)
            cv2.rectangle(img, (kx1, ky1), (kx2, ky2), (255, 0, 0), 2)

        # Save
        out_path = output_dir / f"{file_id}.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
