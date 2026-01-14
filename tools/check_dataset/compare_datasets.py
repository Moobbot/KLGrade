import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
import argparse


def get_image_stats(img_dir, sample_size=500):
    images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
    if len(images) > sample_size:
        images = np.random.choice(images, sample_size, replace=False)

    pixel_values = []
    sharpness_values = []

    print(f"Analyzing {img_dir}...")
    for img_file in tqdm(images):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        pixel_values.extend(img.flatten())
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
        sharpness_values.append(sharpness)

    return np.array(pixel_values), np.array(sharpness_values)


def compare_datasets(
    dir1, dir2, name1="Dataset 1", name2="Dataset 2", output_dir="analysis"
):
    os.makedirs(output_dir, exist_ok=True)

    pixels1, sharp1 = get_image_stats(dir1)
    pixels2, sharp2 = get_image_stats(dir2)

    # Plot Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(pixels1, bins=50, alpha=0.5, label=f"{name1}", density=True, color="blue")
    plt.hist(
        pixels2, bins=50, alpha=0.5, label=f"{name2}", density=True, color="orange"
    )
    plt.title("Pixel Intensity Distribution (CLAHE Check)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "comparison_histogram.png"))
    plt.close()

    # Plot Sharpness
    plt.figure(figsize=(12, 6))
    plt.boxplot([sharp1, sharp2], labels=[name1, name2])
    plt.title("Image Sharpness (Laplacian Variance)")
    plt.savefig(os.path.join(output_dir, "comparison_sharpness.png"))
    plt.close()

    # Report
    print("\n--- Comparison Report ---")
    print(
        f"{name1}: Mean Pixel={pixels1.mean():.2f}, Std={pixels1.std():.2f}, Mean Sharpness={sharp1.mean():.2f}"
    )
    print(
        f"{name2}: Mean Pixel={pixels2.mean():.2f}, Std={pixels2.std():.2f}, Mean Sharpness={sharp2.mean():.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", required=True)
    parser.add_argument("--dir2", required=True)
    parser.add_argument("--name1", default="Dataset 1")
    parser.add_argument("--name2", default="Dataset 2")
    parser.add_argument("--output", default="analysis")

    args = parser.parse_args()
    compare_datasets(args.dir1, args.dir2, args.name1, args.name2, args.output)
