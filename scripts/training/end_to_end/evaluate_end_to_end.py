import torch
import argparse
import sys
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.kiocmil_with_detection import KiocmilWithDetection
from src.datasets.kiocmil_dataset_end_to_end import (
    KiocmilDatasetEndToEnd,
    collate_end_to_end,
)
from src.utils.detection_metrics import compute_iou_metrics


def evaluate_model(
    checkpoint_path: str,
    img_dir: str,
    knee_label_dir: str,
    lesion_label_dir: str,
    split_file: str,
    num_classes: int = 10,
    device: str = "cuda",
    batch_size: int = 16,
    backbone: str = "yolo11l",
):
    print(f"\n{'='*60}")
    print(f"Evaluating End-to-End Model")
    print(f"{'='*60}\n")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {img_dir}")
    print(f"Split: {split_file}")
    print(f"Classes: {num_classes}")

    # Load Model
    print("\n1. Loading Model...")
    model = KiocmilWithDetection(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained_kiocmil=None,
        freeze_kiocmil=False,
    )

    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle state dict structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "N/A")
        print(f"✅ Loaded checkpoint from Epoch {epoch}")
    else:
        state_dict = checkpoint
        print("✅ Loaded state_dict directly")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load Dataset
    print("\n2. Loading Dataset...")
    dataset = KiocmilDatasetEndToEnd(
        img_dir=img_dir,
        knee_label_dir=knee_label_dir,
        lesion_label_dir=lesion_label_dir,
        split_file=split_file,
    )
    print(f"✅ Found {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_end_to_end,
        num_workers=4,
        pin_memory=True,
    )

    # Inference
    all_preds_10 = []
    all_targets = []

    # Detection metrics
    all_pred_knee_boxes = []
    all_gt_knee_boxes = []
    all_pred_lesion_boxes = []
    all_gt_lesion_boxes = []

    print("\n3. Running Inference...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            batch_size = images.shape[0]

            # Get global labels
            global_labels = torch.stack([t["label"] for t in targets]).cpu().numpy()

            # Forward
            # We don't skip classification during eval
            outputs = model(images)

            # Classification predictions
            if "logits_10" in outputs:
                img_logits = outputs["logits_10"]
                # For N-class models, logits_10 contains N logits
                preds = torch.argmax(img_logits, dim=1).cpu().numpy()

                all_preds_10.extend(preds)
                all_targets.extend(global_labels)
            else:
                print("⚠️  Warning: No logits found in output")

            # Detection boxes
            if "knee_boxes" in outputs and "knee_confs" in outputs:
                knee_boxes = outputs["knee_boxes"]  # (B, N, 4)
                knee_confs = outputs["knee_confs"]  # (B, N, 1)

                for b in range(batch_size):
                    # Filter by confidence threshold (0.01 to get any reasonable boxes)
                    conf_mask = knee_confs[b, :, 0] > 0.01
                    pred_boxes = knee_boxes[b][
                        conf_mask
                    ]  # (M, 4) in normalized [cx, cy, w, h]

                    # Convert to [x1, y1, x2, y2] pixel coordinates
                    if len(pred_boxes) > 0:
                        pred_boxes_np = pred_boxes.cpu().numpy()
                        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
                        img_w, img_h = 640, 640  # Assuming square images
                        boxes_xyxy = np.zeros_like(pred_boxes_np)
                        boxes_xyxy[:, 0] = (
                            pred_boxes_np[:, 0] - pred_boxes_np[:, 2] / 2
                        ) * img_w
                        boxes_xyxy[:, 1] = (
                            pred_boxes_np[:, 1] - pred_boxes_np[:, 3] / 2
                        ) * img_h
                        boxes_xyxy[:, 2] = (
                            pred_boxes_np[:, 0] + pred_boxes_np[:, 2] / 2
                        ) * img_w
                        boxes_xyxy[:, 3] = (
                            pred_boxes_np[:, 1] + pred_boxes_np[:, 3] / 2
                        ) * img_h
                        all_pred_knee_boxes.append(boxes_xyxy)
                    else:
                        all_pred_knee_boxes.append(np.array([]))

                    # Ground truth knee boxes from targets
                    if "knees" in targets[b] and len(targets[b]["knees"]) > 0:
                        gt_boxes = []
                        for knee in targets[b]["knees"]:
                            if "ctx_bbox" in knee:
                                bbox = (
                                    knee["ctx_bbox"].cpu().numpy()
                                )  # [cx, cy, w, h] normalized
                                # Convert to [x1, y1, x2, y2]
                                box_xyxy = np.array(
                                    [
                                        (bbox[0] - bbox[2] / 2) * img_w,
                                        (bbox[1] - bbox[3] / 2) * img_h,
                                        (bbox[0] + bbox[2] / 2) * img_w,
                                        (bbox[1] + bbox[3] / 2) * img_h,
                                    ]
                                )
                                gt_boxes.append(box_xyxy)
                        all_gt_knee_boxes.append(
                            np.array(gt_boxes) if gt_boxes else np.array([])
                        )
                    else:
                        all_gt_knee_boxes.append(np.array([]))

    # Metrics
    if len(all_preds_10) == 0:
        print("❌ No predictions made!")
        return None

    print("\n4. Calculating Metrics...")

    # Classification Metrics
    acc = accuracy_score(all_targets, all_preds_10)
    f1_macro = f1_score(all_targets, all_preds_10, average="macro")
    f1_weighted = f1_score(all_targets, all_preds_10, average="weighted")
    cm = confusion_matrix(all_targets, all_preds_10)
    report = classification_report(all_targets, all_preds_10, output_dict=True)
    report_text = classification_report(all_targets, all_preds_10)

    print(f"\n{'='*60}")
    print("CLASSIFICATION METRICS")
    print(f"{'='*60}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 (Macro): {f1_macro:.4f}")
    print(f"✅ F1 (Weighted): {f1_weighted:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report_text)

    # Detection Metrics (IoU)
    detection_metrics = {}
    if all_pred_knee_boxes and all_gt_knee_boxes:

        print(f"\n{'='*60}")
        print("DETECTION METRICS (Knee Boxes)")
        print(f"{'='*60}")

        # Try different IoU thresholds
        for iou_thresh in [0.3, 0.5, 0.7]:
            knee_metrics = compute_iou_metrics(
                all_pred_knee_boxes, all_gt_knee_boxes, iou_threshold=iou_thresh
            )
            detection_metrics[f"knee_iou_{iou_thresh}"] = knee_metrics

            print(f"\nIoU Threshold: {iou_thresh}")
            print(f"  Mean IoU: {knee_metrics['mean_iou']:.4f}")
            print(f"  Precision: {knee_metrics['precision']:.4f}")
            print(f"  Recall: {knee_metrics['recall']:.4f}")
            print(f"  F1: {knee_metrics['f1']:.4f}")
            print(
                f"  TP/FP/FN: {knee_metrics['true_positives']}/{knee_metrics['false_positives']}/{knee_metrics['false_negatives']}"
            )

    results = {
        "classification_metrics": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        },
        "detection_metrics": detection_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End Model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--knee-label-dir", required=True)
    parser.add_argument("--lesion-label-dir", required=True)
    parser.add_argument(
        "--split-file", required=True, help="Path to test/val split txt file"
    )

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--backbone", type=str, default="yolo11l"
    )  # Match training default
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", help="Path to save JSON results")

    args = parser.parse_args()

    results = evaluate_model(
        args.checkpoint,
        args.img_dir,
        args.knee_label_dir,
        args.lesion_label_dir,
        args.split_file,
        num_classes=args.num_classes,
        device=args.device,
        batch_size=args.batch_size,
        backbone=args.backbone,
    )

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
