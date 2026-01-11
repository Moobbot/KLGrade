import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from tqdm import tqdm
import wandb
import os

from src.datasets import (
    CocoDataset,
    create_coco_json,
    detr_collate_fn_dynamic_padding,
)
from src.datasets.transforms import get_detr_processor
from src.datasets.samplers import RepeatFactorSampler
from src.config import CLASSES, CLASSES_10_CLASS, CLASSES_4_CLASS, CLASSES_8_CLASS

# Conditional import for focal loss if it exists in src.losses
try:
    from src.losses import focal_loss_for_detr, calculate_cb_weights_from_coco
except ImportError:
    focal_loss_for_detr = None
    calculate_cb_weights_from_coco = None

class DETRTrainer:
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        output_dir: str,
        model_name: str = "facebook/detr-resnet-50",
        num_classes: int = None,
        use_labels_new: bool = False, # Legacy arg support
        split_dir: str = "splits",
        batch_size: int = 4,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        device: str = None,
        use_balanced_sampler: bool = False,
        use_focal_loss: bool = False,
        wandb_project: str = "KLGrade-Knee-OA"
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = Path(output_dir)
        self.split_dir = split_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_balanced_sampler = use_balanced_sampler
        self.use_focal_loss = use_focal_loss
        self.wandb_project = wandb_project

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine class config
        if num_classes is not None:
            class_map = {
                5: CLASSES,
                10: CLASSES_10_CLASS,
                4: CLASSES_4_CLASS,
                8: CLASSES_8_CLASS,
            }
            self.class_names = class_map.get(num_classes, CLASSES)
            self.num_classes = len(self.class_names)
            self.label_suffix = f"_{num_classes}class"
            self.actual_label_dir = Path(label_dir) # Use as is if explicit
        elif use_labels_new:
            self.class_names = CLASSES_10_CLASS
            self.num_classes = len(CLASSES_10_CLASS)
            self.label_suffix = "_new"
            self.actual_label_dir = Path(label_dir).parent / "labels_new"
        else:
            self.class_names = CLASSES
            self.num_classes = len(CLASSES)
            self.label_suffix = ""
            self.actual_label_dir = Path(label_dir).parent / "labels"
            
        # Fallback if specific subdir logic fails or if user provided exact path
        if not self.actual_label_dir.exists() and Path(label_dir).exists():
             self.actual_label_dir = Path(label_dir)

    def prepare_data(self):
        print("\nStep 1: Preparing COCO annotations...")
        coco_dir = Path("processed/coco")
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_json_path = coco_dir / f"annotations_train{self.label_suffix}.json"
        self.val_json_path = coco_dir / f"annotations_val{self.label_suffix}.json"

        # Create annotations
        create_coco_json(
            yolo_label_dir=str(self.actual_label_dir),
            img_dir=self.img_dir,
            output_path=str(self.train_json_path),
            class_names=self.class_names,
            split_file=str(Path(self.split_dir) / "train.txt"),
        )
        
        create_coco_json(
            yolo_label_dir=str(self.actual_label_dir),
            img_dir=self.img_dir,
            output_path=str(self.val_json_path),
            class_names=self.class_names,
            split_file=str(Path(self.split_dir) / "val.txt"),
        )
        
        # Load Processor
        print("\nStep 2: Loading Processor...")
        self.processor = get_detr_processor(model_name=self.model_name)
        
        # Create Datasets
        print("\nStep 3: Creating Datasets...")
        self.train_dataset = CocoDataset(
            coco_json_path=self.train_json_path, img_dir=self.img_dir, processor=self.processor
        )
        self.val_dataset = CocoDataset(
            coco_json_path=self.val_json_path, img_dir=self.img_dir, processor=self.processor
        )
        
        # Sampler
        train_sampler = None
        shuffle = True
        if self.use_balanced_sampler:
            print("   Using RepeatFactorSampler for balancing...")
            train_sampler = RepeatFactorSampler(self.train_dataset, repeat_thresh=0.1)
            shuffle = False # Sampler handles shuffling
            
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            collate_fn=detr_collate_fn_dynamic_padding,
            num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=detr_collate_fn_dynamic_padding,
            num_workers=2
        )
        
        # Calculate focal loss weights if needed
        self.cb_weights = None
        if self.use_focal_loss and calculate_cb_weights_from_coco:
             print("   Calculating Class-Balanced weights for Focal Loss...")
             self.cb_weights = calculate_cb_weights_from_coco(
                 self.train_json_path, self.num_classes, beta=0.9999
             ).to(self.device)

    def setup_model(self):
        print(f"\nStep 4: Loading Model ({self.model_name})...")
        self.model = DetrForObjectDetection.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self):
        print("\nStep 5: Starting Training...")
        
        run_name = f"DETR-{self.num_classes}class-{self.output_dir.name}"
        if self.use_balanced_sampler: run_name += "-balanced"
        if self.use_focal_loss: run_name += "-focal"
        
        wandb.init(
            project=self.wandb_project,
            name=run_name,
            config={
                "model": self.model_name,
                "num_classes": self.num_classes,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "balanced_sampler": self.use_balanced_sampler,
                "focal_loss": self.use_focal_loss
            }
        )

        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = self.model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                )
                
                if self.use_focal_loss and focal_loss_for_detr and self.cb_weights is not None:
                    # Custom loss calculation
                    loss = focal_loss_for_detr(outputs, labels, self.cb_weights, alpha=0.5, gamma=2.0, num_classes=self.num_classes)
                else:
                    loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation
            avg_val_loss = self.validate(epoch)
            
            self.scheduler.step()
            
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint("best_model.pt", epoch, avg_val_loss)
                
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, avg_val_loss)

        wandb.finish()
        print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = self.model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                )
                
                loss = outputs.loss 
                # Note: Validation typically uses standard loss for comparison unless specifically using focal metric
                # But to compare with train loss, we might want to use same loss function.
                # However, outputs.loss is computed by standard criterion in DETR model.
                # If we used custom loss in training, we might see mismatch. 
                # For simplicity here, we stick to standard loss for validation reporting or use custom if consistent.
                
                val_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        return val_loss / len(self.val_loader)

    def save_checkpoint(self, filename, epoch, val_loss):
        path = self.output_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss
        }, path)
        print(f"  Saved {path}")
