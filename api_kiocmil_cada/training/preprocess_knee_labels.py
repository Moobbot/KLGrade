"""
Preprocess Knee Labels: Map all classes to 0 (Knee) for detection training.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm


def preprocess_labels(input_dir: str, output_dir: str):
    """ "
    Map all class IDs in YOLO label files to 0.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.txt"))
    print(f"Found {len(files)} label files in {input_dir}")

    for file in tqdm(files, desc="Processing labels"):
        with open(file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Set class ID to 0
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")

        with open(output_path / file.name, "w") as f:
            f.writelines(new_lines)

    print(f"\n✅  Processed labels saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input label directory")
    parser.add_argument("--output", required=True, help="Output label directory")
    args = parser.parse_args()

    preprocess_labels(args.input, args.output)
