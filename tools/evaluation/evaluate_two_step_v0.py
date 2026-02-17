import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from api_two_step_yolo.inference.two_step_yolo_api import TwoStepYOLOInference
import argparse


def get_gt_kl_grade(label_path):
    """
    Reads YOLO label file and determines Image-level KL Grade.
    Assumption: Classes 0-4 correspond to KL0-KL4.
    Image Grade = Max(Class IDs).
    If file parsing fails or empty, returns 0.
    """
    if not os.path.exists(label_path):
        return 0

    classes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(float(parts[0]))
                    classes.append(cls_id)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return 0

    if not classes:
        return 0

    # Filter classes to valid range 0-4
    valid_classes = [c for c in classes if 0 <= c <= 4]

    if not valid_classes:
        return 0

    return max(valid_classes)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Two-Step Pipeline on Dataset V0"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/dataset_v0",
        help="Path to dataset_v0",
    )
    parser.add_argument(
        "--knee-model", type=str, default="runs/detect/knee_detector/weights/best.pt"
    )
    parser.add_argument(
        "--lesion-model",
        type=str,
        default="runs/detect/lesion_8class_balanced/weights/best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation_v0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing Pipeline...")
    pipeline = TwoStepYOLOInference(
        knee_model_path=args.knee_model,
        lesion_model_path=args.lesion_model,
        device=args.device,
        knee_conf_threshold=0.75,
        lesion_conf_threshold=0.25,
    )

    image_files = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    # Filter for image extensions
    image_files = [
        f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_files)} images.")

    results = []

    for img_path in tqdm(image_files):
        # Determine GT path
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(args.dataset_dir, "labels", name_without_ext + ".txt")

        gt_grade = get_gt_kl_grade(label_path)

        # Inference
        try:
            pred = pipeline.predict(img_path)
            pred_grade = pred["kl_grade"]

            # Handle None prediction cases
            if pred_grade is None:
                pred_grade = 0  # Default to 0 if detection failed? Or keep None?
                # Pipeline returns None if no *knees* detected.
                # If knee detected but no lesions -> KL0.
                # If no knee -> treated as prediction failure.
                # Let's count as error or KL0?
                # For evaluation, if no knee was found, we can arguably say we predicted KL0 (healthy/nothing found)
                # or mark it as failure. Let's map to 0 to compute metrics, but track 'detected' status.

            results.append(
                {
                    "image": basename,
                    "gt_grade": gt_grade,
                    "pred_grade": pred_grade,
                    "knees_detected": len(pred["knees"]),
                    "lesions_detected": len(pred["lesions"]),
                }
            )

        except Exception as e:
            print(f"Error processing {basename}: {e}")

    # Analysis
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    # Calculate Metrics
    y_true = df["gt_grade"]
    y_pred = df["pred_grade"]

    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print("\n" + "=" * 40)
    print(f"Evaluation Results (N={len(df)})")
    print("=" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"MSE:      {mse:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    print("\nConfusion Matrix:")
    print(cm)

    # Save CM plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1, 2, 3, 4],
        yticklabels=[0, 1, 2, 3, 4],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix - Two Step Pipeline")
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
