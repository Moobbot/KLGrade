import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import KFold

import torch
from torch import nn, optim

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dataset import ImageDataset
from early_stop import EarlyStopping
from model import model_return
from config import BATCH_SIZE, EPOCHS, K_FOLDS

# Thêm import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def collate_fn(batch):
    # Nếu là detection (target là dict), trả về tuple
    if isinstance(batch[0]["target"], dict):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "target": {
                "boxes": [item["target"]["boxes"] for item in batch],
                "labels": [item["target"]["labels"] for item in batch],
            },
        }
    else:
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "target": torch.stack([item["target"] for item in batch]),
        }


def train_for_kfold(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    fold,
    epoch,
    save_metrics_path=None,
):
    train_loss = 0.0
    cls_loss = 0.0
    loc_loss = 0.0
    total_loss = 0.0
    metric = None
    # Nếu là detection, khởi tạo metric
    is_detection = hasattr(model, "roi_heads")
    if is_detection:
        metric = MeanAveragePrecision(class_metrics=True).to("cuda")
    model.train()
    with torch.set_grad_enabled(True):
        for batch in tqdm(
            dataloader, desc=f"Fold {fold} Epoch {epoch} Train", unit="Batch"
        ):
            optimizer.zero_grad()
            image = batch["image"].cuda()
            target = batch["target"]
            if is_detection:
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                # import IPython; IPython.embed()
                loss = sum(loss for loss in loss_dict.values())
                train_loss += loss.item()
                cls_loss += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                loc_loss += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                total_loss += loss.item()
                # Lấy predict để update metric
                model.eval()
                pred = model(image)
                if isinstance(pred, tuple):
                    pred_logits, pred_boxes = pred
                    preds = []
                    for s, b in zip(pred_logits, pred_boxes):
                        preds.append(
                            {
                                "scores": s.detach().cuda(),
                                "boxes": b.detach().cuda(),
                                "labels": torch.zeros_like(s, dtype=torch.long).cuda(),
                            }
                        )
                else:
                    preds = [
                        {
                            "boxes": p["boxes"].detach().cuda(),
                            "scores": p["scores"].detach().cuda(),
                            "labels": p["labels"].detach().cuda(),
                        }
                        for p in pred
                    ]
                gts = [
                    {
                        "boxes": t["boxes"].detach().cuda(),
                        "labels": t["labels"].detach().cuda(),
                    }
                    for t in targets
                ]
                metric.update(preds, gts)
                model.train()
            else:
                labels = batch["target"].cuda()
                output = model(image)
                loss = criterion(output, labels)
                train_loss += loss.item()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    metrics_result = None
    if is_detection:
        metrics_result = metric.compute()
        if save_metrics_path is not None:
            with open(save_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"Fold {fold} Epoch {epoch} Train Detection Metrics:\n")
                f.write(
                    f"  cls_loss: {cls_loss:.4f}, loc_loss: {loc_loss:.4f}, total_loss: {total_loss:.4f}\n"
                )
                f.write(
                    f"  mAP: {metrics_result['map']:.4f}, Precision: {metrics_result['map_50']:.4f}, Recall: {metrics_result['mar_100']:.4f}\n"
                )
                for i, ap in enumerate(metrics_result["map_per_class"]):
                    f.write(f"    Class {i} AP: {ap:.4f}\n")
                f.write("\n")
    return train_loss, cls_loss, loc_loss, total_loss, metrics_result


def val_for_kfold(model, dataloader, criterion, fold, epoch, save_metrics_path=None):
    val_loss = 0.0
    cls_loss = 0.0
    loc_loss = 0.0
    total_loss = 0.0
    # Khởi tạo metric detection nếu là detection
    metric = MeanAveragePrecision(class_metrics=True).to("cuda")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Fold {fold} Epoch {epoch} Valid", unit="Batch"
        ):
            image = batch["image"].cuda()
            target = batch["target"]
            if hasattr(model, "roi_heads"):  # Detection
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
                cls_loss += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                loc_loss += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                total_loss += loss.item()
                # Lấy predict để update metric
                model.eval()
                pred = model(image)
                # pred: list of dicts with 'boxes', 'labels', 'scores'
                # targets: list of dicts
                # Đảm bảo pred và targets đúng format cho torchmetrics
                if isinstance(pred, tuple):
                    # Nếu model trả về (scores, boxes)
                    pred_logits, pred_boxes = pred
                    preds = []
                    for s, b in zip(pred_logits, pred_boxes):
                        preds.append(
                            {
                                "scores": s.detach().cuda(),
                                "boxes": b.detach().cuda(),
                                "labels": torch.zeros_like(
                                    s, dtype=torch.long
                                ).cuda(),  # dummy nếu không có labels
                            }
                        )
                else:
                    preds = [
                        {
                            "boxes": p["boxes"].detach().cuda(),
                            "scores": p["scores"].detach().cuda(),
                            "labels": p["labels"].detach().cuda(),
                        }
                        for p in pred
                    ]
                gts = [
                    {
                        "boxes": t["boxes"].detach().cuda(),
                        "labels": t["labels"].detach().cuda(),
                    }
                    for t in targets
                ]
                metric.update(preds, gts)
            else:
                labels = batch["target"].cuda()
                output = model(image)
                loss = criterion(output, labels)
                val_loss += loss.item()
    # Tính metric detection nếu là detection
    metrics_result = None
    if hasattr(model, "roi_heads"):
        metrics_result = metric.compute()
        if save_metrics_path is not None:
            with open(save_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"Fold {fold} Epoch {epoch} Detection Metrics:\n")
                f.write(
                    f"  cls_loss: {cls_loss:.4f}, loc_loss: {loc_loss:.4f}, total_loss: {total_loss:.4f}\n"
                )
                f.write(
                    f"  mAP: {metrics_result['map']:.4f}, Precision: {metrics_result['map_50']:.4f}, Recall: {metrics_result['mar_100']:.4f}\n"
                )
                for i, ap in enumerate(metrics_result["map_per_class"]):
                    f.write(f"    Class {i} AP: {ap:.4f}\n")
                f.write("\n")
    return val_loss, cls_loss, loc_loss, total_loss, metrics_result


def train(train_dataset, val_dataset, args, batch_size, epochs):
    # DataLoader cho toàn bộ train/val
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model_ft = model_return(args)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Loss Function
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01
    )  # Optimizer
    scheduler = None

    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft.cuda()

    history = {"train_loss": [], "val_loss": []}

    patience = 7
    delta = 0.1
    early_stopping = EarlyStopping(args, patience=patience, verbose=True, delta=delta)

    metric = None
    is_detection = hasattr(model_ft, "roi_heads")
    if is_detection:
        metric = MeanAveragePrecision(class_metrics=True).to("cuda")

    for epoch in range(1, epochs + 1):
        if epoch == 2:
            for param in model_ft.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model_ft.parameters()),
                weight_decay=0.0001,
                lr=0.001,
            )
            scheduler = MultiStepLR(optimizer, milestones=[2], gamma=0.1)

        print(f"Learning Rate : {optimizer.param_groups[0]['lr']}")

        train_loss, train_cls_loss, train_loc_loss, train_total_loss, train_metrics = (
            train_for_kfold(
                model_ft, train_loader, criterion, optimizer, scheduler, 1, epoch
            )
        )
        val_loss, val_cls_loss, val_loc_loss, val_total_loss, val_metrics = (
            val_for_kfold(model_ft, val_loader, criterion, 1, epoch)
        )

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        print(
            f"Epoch: {epoch}/{epochs} \t Avg Train Loss: {train_loss:.3f} \t Avg Valid Loss: {val_loss:.3f}"
        )
        if train_metrics is not None:
            print(
                f"  Train mAP: {train_metrics['map']:.4f}, Precision: {train_metrics['map_50']:.4f}, Recall: {train_metrics['mar_100']:.4f}"
            )
            for i, ap in enumerate(train_metrics["map_per_class"]):
                print(f"    Train Class {i} AP: {ap:.4f}")
        if val_metrics is not None:
            print(
                f"  Val mAP: {val_metrics['map']:.4f}, Precision: {val_metrics['map_50']:.4f}, Recall: {val_metrics['mar_100']:.4f}"
            )
            for i, ap in enumerate(val_metrics["map_per_class"]):
                print(f"    Val Class {i} AP: {ap:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        early_stopping(val_loss, model_ft, args, 1, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print()
    print(f"Performance of training:")
    print(
        f"Avg Train Loss: {np.mean(history['train_loss']):.3f} \t Avg Valid Loss: {np.mean(history['val_loss']):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", dest="model_type", action="store")
    parser.add_argument(
        "-i", "--image_size", type=int, default=224, dest="image_size", action="store"
    )
    args = parser.parse_args()

    image_size_tuple = (args.image_size, args.image_size)

    print(f"Model Type : {args.model_type}")
    print(f"Image Size : {image_size_tuple}")

    train_csv = pd.read_csv("./dataset/dataset.csv")
    train_transform = A.Compose(
        [
            A.Resize(
                args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1
            ),
            # A.HorizontalFlip(p=0.5),
            # A.Rotate(limit=20, p=1),
            # A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.01
        ),
    )
    val_transform = A.Compose(
        [
            A.Resize(
                args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1
            ),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.01
        ),
    )
    train_dataset = ImageDataset(
        train_csv, transforms=train_transform, return_torchvision=True
    )
    val_dataset = ImageDataset(
        train_csv, transforms=val_transform, return_torchvision=True
    )

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    torch.manual_seed(42)

    train(
        train_dataset,
        val_dataset,
        args,
        batch_size,
        epochs,
    )
