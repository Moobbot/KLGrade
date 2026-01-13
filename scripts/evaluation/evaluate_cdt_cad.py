"""
Evaluation script for CDT-CAD model

Usage:
    python scripts/evaluation/evaluate_cdt_cad.py --model runs/cdt_cad/best.pt --data processed/yolo11_labels.yaml
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import json
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.cdt_cad import CDTCAD
from src.datasets.cdt_cad_dataset import CDTCADDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CDT-CAD model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data YAML file"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.01, help="Confidence threshold"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]

    # Create model
    model = CDTCAD(
        **{
            "num_classes": config["model"]["num_classes"],
            "num_queries": config["model"].get("num_queries", 100),
            "hidden_dim": config["model"].get("hidden_dim", 256),
            "num_encoder_layers": config["model"].get("num_encoder_layers", 6),
            "num_decoder_layers": config["model"].get("num_decoder_layers", 6),
            "num_feature_levels": config["model"].get("num_feature_levels", 4),
            "n_heads": config["model"].get("n_heads", 8),
            "dim_feedforward": config["model"].get("dim_feedforward", 1024),
            "dropout": config["model"].get("dropout", 0.1),
            "n_points": config["model"].get("n_points", 4),
            "dilation_rates": config["feature_extractor"].get(
                "dilation_rates", [1, 2, 4, 8]
            ),
            "num_iterations": config["feature_extractor"].get("num_iterations", 3),
            "wavelet": config["feature_extractor"].get("wavelet_type", "haar"),
            "pretrained_backbone": False,
        }
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, config


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.01,
):
    """
    Evaluate model and generate COCO-format predictions
    """
    predictions = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for images, targets in pbar:
        images = images.to(device)

        # Forward
        outputs = model(images)

        # Process predictions
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes+1]
        pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]

        # Get probabilities and filter by confidence
        prob = pred_logits.softmax(-1)  # [B, num_queries, num_classes+1]
        scores, labels = prob[..., :-1].max(-1)  # Exclude "no object" class

        # Process each image in batch
        for i, target in enumerate(targets):
            img_h, img_w = 800, 800  # Original image size (before normalization)

            # Filter by confidence
            keep = scores[i] > conf_threshold

            scores_keep = scores[i][keep].cpu().numpy()
            labels_keep = labels[i][keep].cpu().numpy()
            boxes_keep = pred_boxes[i][keep].cpu().numpy()

            # Convert boxes from (cx, cy, w, h) normalized to (x, y, w, h) absolute
            for score, label, box in zip(scores_keep, labels_keep, boxes_keep):
                cx, cy, w, h = box

                # Denormalize and convert to COCO format (x, y, w, h)
                x = (cx - w / 2) * img_w
                y = (cy - h / 2) * img_h
                w = w * img_w
                h = h * img_h

                predictions.append(
                    {
                        "image_id": target["image_id"].item(),
                        "category_id": int(label),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                    }
                )

    return predictions


def create_coco_format(dataset: CDTCADDataset, num_classes: int):
    """Create COCO-format ground truth annotations"""
    images = []
    annotations = []
    categories = []

    # Categories
    class_names = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    for i in range(num_classes):
        categories.append(
            {
                "id": i,
                "name": class_names[i] if i < len(class_names) else f"class_{i}",
                "supercategory": "knee",
            }
        )

    # Images and annotations
    ann_id = 0
    for img_id in range(len(dataset)):
        _, target = dataset[img_id]

        images.append(
            {
                "id": img_id,
                "width": 800,
                "height": 800,
                "file_name": dataset.image_names[img_id],
            }
        )

        # Convert boxes and add annotations
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        for box, label in zip(boxes, labels):
            cx, cy, w, h = box

            # Convert to COCO format (x, y, w, h) absolute
            x = (cx - w / 2) * 800
            y = (cy - h / 2) * 800
            w = w * 800
            h = h * 800

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return coco_format


def compute_coco_metrics(gt_coco, predictions):
    """Compute COCO evaluation metrics"""
    # Create predictions in COCO format
    coco_dt = gt_coco.loadRes(predictions) if predictions else COCO()

    # Evaluate
    coco_eval = COCOeval(gt_coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        "mAP": coco_eval.stats[0],
        "mAP50": coco_eval.stats[1],
        "mAP75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

    return metrics


def main():
    args = parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.model).parent / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, config = load_model(args.model, device)
    print(f"Model loaded from {args.model}")

    # Load data config
    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)

    # Create validation dataset
    print("Loading validation dataset...")
    val_img_dir = data_config["val"]
    label_dir = str(Path(val_img_dir).parent / "labels")
    val_split = (
        Path("processed/splits/val.txt")
        if Path("processed/splits/val.txt").exists()
        else None
    )

    val_dataset = CDTCADDataset(
        img_dir=val_img_dir,
        label_dir=label_dir,
        split_file=val_split,
        image_size=(800, 800),
        augment=False,
        clahe=False,
        num_classes=config["model"]["num_classes"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Validation set: {len(val_dataset)} images")

    # Evaluate
    print("\nRunning evaluation...")
    predictions = evaluate(model, val_loader, device, args.conf_threshold)

    print(f"Generated {len(predictions)} predictions")

    # Save predictions
    pred_file = output_dir / "predictions.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {pred_file}")

    # Create COCO ground truth
    print("\nCreating COCO ground truth...")
    gt_coco_dict = create_coco_format(val_dataset, config["model"]["num_classes"])

    gt_file = output_dir / "ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(gt_coco_dict, f, indent=2)
    print(f"Ground truth saved to {gt_file}")

    # Compute metrics
    print("\nComputing COCO metrics...")
    gt_coco = COCO(str(gt_file))

    if predictions:
        metrics = compute_coco_metrics(gt_coco, predictions)

        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"mAP (IoU=0.50:0.95): {metrics['mAP']:.4f}")
        print(f"mAP50 (IoU=0.50):    {metrics['mAP50']:.4f}")
        print(f"mAP75 (IoU=0.75):    {metrics['mAP75']:.4f}")
        print(f"AR@100:              {metrics['AR@100']:.4f}")
        print("=" * 50)

        # Save summary
        with open(output_dir / "results.txt", "w") as f:
            f.write("CDT-CAD Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.data}\n")
            f.write(f"Confidence Threshold: {args.conf_threshold}\n\n")
            f.write("Metrics:\n")
            f.write(f"  mAP (IoU=0.50:0.95): {metrics['mAP']:.4f}\n")
            f.write(f"  mAP50 (IoU=0.50):    {metrics['mAP50']:.4f}\n")
            f.write(f"  mAP75 (IoU=0.75):    {metrics['mAP75']:.4f}\n")
            f.write(f"  AR@100:              {metrics['AR@100']:.4f}\n")

        print(f"\nResults saved to {output_dir}")
    else:
        print("No predictions generated. Check confidence threshold.")


if __name__ == "__main__":
    main()
