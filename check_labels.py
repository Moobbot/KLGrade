import os

labels_dir = "dataset/labels"


def check_invalid_boxes(labels_dir):
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            print(fname)
            continue
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, "r") as f:
            for idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"[{fname}][Line {idx}] Format error: {line.strip()}")
                    continue
                try:
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    print(f"[{fname}][Line {idx}] Value error: {line.strip()}")
                    continue
                if width <= 0 or height <= 0:
                    print(
                        f"[{fname}][Line {idx}] Invalid box: width={width}, height={height} | {line.strip()}"
                    )


if __name__ == "__main__":
    check_invalid_boxes(labels_dir)
