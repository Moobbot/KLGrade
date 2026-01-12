import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import sys
import os
import wandb
import numpy as np
from collections import Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.kiocmil_dataset import KiocmilDataset, collate_kiocmil
from src.models.kiocmil_model import KiocmilModel
from src.config import PROJECT_ROOT


class KiocmilTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # WandB Init
        if not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
            )

        # Data
        print("Initializing Datasets...")
        self.train_dataset = KiocmilDataset(
            img_dir=args.img_dir,
            knee_label_dir=args.knee_labels,
            lesion_label_dir=args.lesion_labels,
            split_file=args.train_split,
            transform=None,  # Add augmentation later if needed
            ctx_size=(384, 384),
            patch_size=(224, 224),
        )
        self.val_dataset = KiocmilDataset(
            img_dir=args.img_dir,
            knee_label_dir=args.knee_labels,
            lesion_label_dir=args.lesion_labels,
            split_file=args.val_split,
            transform=None,
            ctx_size=(384, 384),
            patch_size=(224, 224),
        )

        # Sampler / Shuffle Logic
        sampler = None
        shuffle = True

        # Calculate class weights for Sampler or Loss
        # Get labels from dataset
        labels = [
            self.train_dataset.labels_map.get(f, 0)
            for f in self.train_dataset.image_files
        ]
        self.class_counts = Counter(labels)
        print(f"Class Distribution: {dict(self.class_counts)}")

        num_classes = 10
        # sort counts by class id 0-9
        counts_list = [self.class_counts.get(i, 0) for i in range(num_classes)]

        # Avoid zero division
        counts_list = [c if c > 0 else 1 for c in counts_list]

        # Raw weights: 1/count
        class_weights_raw = [1.0 / c for c in counts_list]

        # Normalize weights for Loss (sum = num_classes) -> optional, helps keeps scale
        sum_w = sum(class_weights_raw)
        self.class_weights_norm = (
            torch.tensor([w * num_classes / sum_w for w in class_weights_raw])
            .float()
            .to(self.device)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_kiocmil,
            num_workers=4,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_kiocmil,
            num_workers=4,
        )

        # Model
        print("Initializing Model...")
        self.model = KiocmilModel(backbone_name=args.backbone).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=1e-4
        )

        # Criterions
        self.crit_10 = nn.CrossEntropyLoss()
        self.crit_grade = nn.CrossEntropyLoss()
        self.crit_type = nn.BCEWithLogitsLoss()

    def compute_loss(self, outputs, target_10):
        # target_10: (B) values 0-9

        # Derived targets
        target_grade = target_10 // 2  # 0-4
        target_type = (
            (target_10 % 2 == 0).float().unsqueeze(1)
        )  # Even=Osteophyte(1)? No check spec.
        # Object-Context.txt: "type_ost = 1 nếu y10 chẵn (a), 0 nếu lẻ (b)"
        # Wait, usually a (0) is Ost, b (1) is JS.
        # Just stick to User Spec:
        # "type_ost = 1 nếu y10 chẵn (a)" -> "if y10 % 2 == 0: 1 else 0"

        loss_10 = self.crit_10(outputs["logits_10"], target_10)
        loss_grade = self.crit_grade(outputs["logits_grade"], target_grade)
        loss_type = self.crit_type(outputs["logits_type"], target_type)

        total_loss = loss_10 + 0.5 * loss_grade + 0.25 * loss_type
        return total_loss, {
            "l10": loss_10.item(),
            "lgrade": loss_grade.item(),
            "ltype": loss_type.item(),
        }

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")

        for batch in pbar:
            # Prepare Data involves moving tensors to device inside the list of dicts
            # collate_kiocmil just returns list of dicts.
            # We need to manually move tensors to device.

            # Extract Dummy Labels (since dataset returns 0)
            # In real usage, this must be from batch['label_10']
            # We create dummy random targets if all are 0 (debug mode)
            # or trust the dataset. Use dataset values.

            label_10_list = [item["label_10"] for item in batch]
            targets = torch.tensor(label_10_list).long().to(self.device)

            # Move images to device
            for item in batch:
                for knee in item["knees"]:
                    knee["ctx"] = knee["ctx"].to(self.device)
                    knee["js"] = knee["js"].to(self.device)
                    knee["ost"] = knee["ost"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch)

            loss, loss_dict = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if not self.args.no_wandb:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        **{f"train_{k}": v for k, v in loss_dict.items()},
                    }
                )

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct_10 = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                label_10_list = [item["label_10"] for item in batch]
                targets = torch.tensor(label_10_list).long().to(self.device)

                for item in batch:
                    for knee in item["knees"]:
                        knee["ctx"] = knee["ctx"].to(self.device)
                        knee["js"] = knee["js"].to(self.device)
                        knee["ost"] = knee["ost"].to(self.device)

                outputs = self.model(batch)
                loss, _ = self.compute_loss(outputs, targets)
                total_loss += loss.item()

                # Accuracy
                preds = torch.argmax(outputs["logits_10"], dim=1)
                correct_10 += (preds == targets).sum().item()
                total += targets.size(0)

        acc = correct_10 / total if total > 0 else 0
        return total_loss / len(self.val_loader), acc

    def run(self):
        best_acc = 0
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )

            if not self.args.no_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "epoch_train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
                print("Saved Best Model")

            # Always save last
            torch.save(self.model.state_dict(), self.save_dir / "last_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="dataset/dataset_v0/images")
    parser.add_argument(
        "--knee_labels", type=str, default="dataset/dataset_v0/labels-knee"
    )
    parser.add_argument(
        "--lesion_labels", type=str, default="dataset/dataset_v0/labels_new"
    )
    parser.add_argument("--train_split", type=str, default="splits/train.txt")
    parser.add_argument("--val_split", type=str, default="splits/val.txt")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--save_dir", type=str, default="runs/kiocmil_exp1")
    parser.add_argument("--batch_size", type=int, default=2)  # Small batch for MIL
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    # WandB configs
    parser.add_argument("--wandb_project", type=str, default="KIOCMIL_Project")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    args = parser.parse_args()
    trainer = KiocmilTrainer(args)
    trainer.run()
