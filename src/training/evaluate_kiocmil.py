import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.kiocmil_dataset import KiocmilDataset, collate_kiocmil
from src.models.kiocmil_model import KiocmilModel
from src.config import PROJECT_ROOT


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print(f"Loading dataset from {args.split_file}...")
    dataset = KiocmilDataset(
        img_dir=args.img_dir,
        knee_label_dir=args.knee_labels,
        lesion_label_dir=args.lesion_labels,
        split_file=args.split_file,
        transform=None,
        ctx_size=(384, 384),
        patch_size=(224, 224),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_kiocmil,
        num_workers=4,
    )

    # Model
    print(f"Loading model from {args.model_path}...")
    model = KiocmilModel(backbone_name=args.backbone).to(device)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Metrics
    all_preds_10 = []
    all_targets_10 = []
    all_preds_grade = []
    all_targets_grade = []
    all_preds_type = []
    all_targets_type = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move data to device
            label_10_list = [item["label_10"] for item in batch]
            targets = torch.tensor(label_10_list).long().to(device)

            for item in batch:
                for knee in item["knees"]:
                    knee["ctx"] = knee["ctx"].to(device)
                    knee["js"] = knee["js"].to(device)
                    knee["ost"] = knee["ost"].to(device)

            # Forward
            outputs = model(batch)

            # Predictions
            logits_10 = outputs["logits_10"]
            preds_10 = torch.argmax(logits_10, dim=1)

            # 10-class (0-9)
            all_preds_10.extend(preds_10.cpu().numpy())
            all_targets_10.extend(targets.cpu().numpy())

            # Derived Grade (0-4)
            logits_grade = outputs["logits_grade"]
            preds_grade = torch.argmax(logits_grade, dim=1)
            targets_grade = targets // 2

            all_preds_grade.extend(preds_grade.cpu().numpy())
            all_targets_grade.extend(targets_grade.cpu().numpy())

            # Derived Type (0 or 1)
            # Use User Spec: "type_ost = 1 if y10 is even (a), 0 if odd (b)" ??
            # Wait, config.py says: even=Osteophyte (a), odd=JointSpace (b)
            # KiocmilTrainer logic: "target_type = (target_10 % 2 == 0)" -> 1 if even
            # So 1=Osteophyte, 0=JointSpace?
            # Let's verify Model Output Logic.
            # outputs['logits_type'] is 1 dim.
            # sigmoid(logits) > 0.5 -> 1.

            logits_type = outputs["logits_type"]
            probs_type = torch.sigmoid(logits_type)
            preds_type = (probs_type > 0.5).long().view(-1)
            targets_type = (targets % 2 == 0).long()  # 1 if even

            all_preds_type.extend(preds_type.cpu().numpy())
            all_targets_type.extend(targets_type.cpu().numpy())

    # Compute Metrics
    acc_10 = metrics.accuracy_score(all_targets_10, all_preds_10)
    acc_grade = metrics.accuracy_score(all_targets_grade, all_preds_grade)
    acc_type = metrics.accuracy_score(all_targets_type, all_preds_type)

    print("\n" + "=" * 40)
    print(" EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Samples: {len(all_targets_10)}")
    print(f"Accuracy (10-class): {acc_10:.4f}")
    print(f"Accuracy (5-class Grade): {acc_grade:.4f}")
    print(f"Accuracy (Type Detection): {acc_type:.4f}")

    print("\nClassification Report (10-class):")
    print(metrics.classification_report(all_targets_10, all_preds_10))

    print("\nConfusion Matrix (5-class Grade):")
    print(metrics.confusion_matrix(all_targets_grade, all_preds_grade))

    # Save details if needed (TODO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="dataset/dataset_v0/images")
    parser.add_argument(
        "--knee_labels", type=str, default="dataset/dataset_v0/labels-knee"
    )
    parser.add_argument(
        "--lesion_labels", type=str, default="dataset/dataset_v0/labels_10_class"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to split file (e.g. splits/val.txt)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained .pth model"
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    evaluate(args)
