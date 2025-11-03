"""
Integrate pre-generated synthetic images into the classification CSV.

Expected folder structure for generated images:
  processed/generative/
    KL0/*.jpg
    KL1/*.jpg
    KL2/*.jpg
    KL3/*.jpg
    KL4/*.jpg

This script scans the above directories and appends entries to
processed/classification/labels.csv with the corresponding class id.

If you have your own generative pipeline (e.g., diffusion, GAN), export images
into those class folders and run this script to include them in training.
"""
import os
import sys
import csv
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES


def parse_args():
    p = argparse.ArgumentParser(description="Integrate synthetic images into labels.csv")
    p.add_argument("--gen_dir", default=os.path.join("processed", "generative"), help="Root directory of generated images")
    p.add_argument("--csv", default=os.path.join("processed", "classification", "labels.csv"), help="Existing classification CSV to append to")
    p.add_argument("--out_csv", default=os.path.join("processed", "classification", "labels_with_generated.csv"), help="Output CSV with generated data appended")
    return p.parse_args()


def main():
    args = parse_args()
    class_to_id = {v: k for k, v in CLASSES.items()}
    rows = []

    # read original
    if os.path.exists(args.csv):
        with open(args.csv, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) >= 2:
                    rows.append([row[0], int(row[1])])

    # append generated
    for cid, cname in CLASSES.items():
        cdir = os.path.join(args.gen_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append([os.path.join(cdir, fn), cid])

    # write out
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for p, c in rows:
            w.writerow([p, c])
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
