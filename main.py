import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch import nn, optim

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from utils import ImageDataset, make_dataset_dataframe
from early_stop import EarlyStopping
from model import model_return
from config import BATCH_SIZE, EPOCHS

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


def sum_dict_scalars(d):
    return sum(
        v.item() if torch.is_tensor(v) and v.dim() == 0 else v
        for v in d.values()
        if (isinstance(v, (int, float)) or (torch.is_tensor(v) and v.dim() == 0))
    )


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, epoch, save_metrics_path=None
):
    train_loss = 0.0
    cls_loss = 0.0
    loc_loss = 0.0
    total_loss = 0.0
    metric = None
    is_detection = hasattr(model, "roi_heads")
    if is_detection:
        metric = MeanAveragePrecision(class_metrics=True).to("cuda")
        device = next(model.parameters()).device
    model.train()
    with torch.set_grad_enabled(True):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} Train", unit="Batch"):
            optimizer.zero_grad()
            image = batch["image"].cuda()
            target = batch["target"]
            if is_detection:
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                if isinstance(loss_dict, dict):
                    loss = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, list):
                    if all(isinstance(item, dict) for item in loss_dict):
                        loss = sum(sum_dict_scalars(d) for d in loss_dict)
                    else:
                        loss = sum(loss_dict)
                else:
                    loss = loss_dict
                train_loss += loss.item() if torch.is_tensor(loss) else loss
                cls_loss += (
                    loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                    if isinstance(loss_dict, dict)
                    else 0.0
                )
                loc_loss += (
                    loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                    if isinstance(loss_dict, dict)
                    else 0.0
                )
                total_loss += loss.item() if torch.is_tensor(loss) else loss
                # Lấy predict để update metric
                model.eval()
                pred = model(image)
                if isinstance(pred, tuple):
                    pred_logits, pred_boxes = pred
                    preds = []
                    for s, b in zip(pred_logits, pred_boxes):
                        preds.append(
                            {
                                "scores": s.detach().to(device),
                                "boxes": b.detach().to(device),
                                "labels": torch.zeros_like(s, dtype=torch.long).to(
                                    device
                                ),
                            }
                        )
                else:
                    preds = [
                        {
                            "boxes": p["boxes"].detach().to(device),
                            "scores": p["scores"].detach().to(device),
                            "labels": p["labels"].detach().to(device),
                        }
                        for p in pred
                    ]
                gts = [
                    {
                        "boxes": t["boxes"].detach().to(device),
                        "labels": t["labels"].detach().to(device),
                    }
                    for t in targets
                ]
                metric.update(preds, gts)
                model.train()
            else:
                if isinstance(batch["target"], torch.Tensor):
                    labels = batch["target"].cuda()
                else:
                    # If it's a dict, move each value to cuda
                    labels = {k: v.cuda() for k, v in batch["target"].items()}
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
                f.write(f"Epoch {epoch} Train Detection Metrics:\n")
                f.write(
                    f"cls_loss: {cls_loss:.4f}, loc_loss: {loc_loss:.4f}, total_loss: {total_loss:.4f}\n"
                )
                f.write(
                    f"mAP: {metrics_result['map']:.4f}, Precision: {metrics_result['map_50']:.4f}, Recall: {metrics_result['mar_100']:.4f}\n"
                )
                for i, ap in enumerate(metrics_result["map_per_class"]):
                    f.write(f"Class {i} AP: {ap:.4f}\n")
                f.write("\n")
    return train_loss, cls_loss, loc_loss, total_loss, metrics_result


def val_one_epoch(model, dataloader, criterion, epoch, save_metrics_path=None):
    val_loss = 0.0
    cls_loss = 0.0
    loc_loss = 0.0
    total_loss = 0.0
    metric = None
    is_detection = hasattr(model, "roi_heads")
    if is_detection:
        metric = MeanAveragePrecision(class_metrics=True).to("cuda")
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} Valid", unit="Batch"):
            image = batch["image"].cuda()
            target = batch["target"]
            if is_detection:
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                if isinstance(loss_dict, dict):
                    loss = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, list):
                    if all(isinstance(item, dict) for item in loss_dict):
                        loss = sum(sum_dict_scalars(d) for d in loss_dict)
                    else:
                        loss = sum(loss_dict)
                else:
                    loss = loss_dict
                val_loss += loss.item() if torch.is_tensor(loss) else loss
                cls_loss += (
                    loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                    if isinstance(loss_dict, dict)
                    else 0.0
                )
                loc_loss += (
                    loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                    if isinstance(loss_dict, dict)
                    else 0.0
                )
                total_loss += loss.item() if torch.is_tensor(loss) else loss
                model.eval()
                pred = model(image)
                if isinstance(pred, tuple):
                    pred_logits, pred_boxes = pred
                    preds = []
                    for s, b in zip(pred_logits, pred_boxes):
                        preds.append(
                            {
                                "scores": s.detach().to(device),
                                "boxes": b.detach().to(device),
                                "labels": torch.zeros_like(s, dtype=torch.long).to(
                                    device
                                ),
                            }
                        )
                else:
                    preds = [
                        {
                            "boxes": p["boxes"].detach().to(device),
                            "scores": p["scores"].detach().to(device),
                            "labels": p["labels"].detach().to(device),
                        }
                        for p in pred
                    ]
                gts = [
                    {
                        "boxes": t["boxes"].detach().to(device),
                        "labels": t["labels"].detach().to(device),
                    }
                    for t in targets
                ]
                metric.update(preds, gts)
            else:
                labels = batch["target"].cuda()
                output = model(image)
                loss = criterion(output, labels)
                val_loss += loss.item()
    metrics_result = None
    if is_detection:
        metrics_result = metric.compute()
        if save_metrics_path is not None:
            with open(save_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch} Detection Metrics:\n")
                f.write(
                    f"cls_loss: {cls_loss:.4f}, loc_loss: {loc_loss:.4f}, total_loss: {total_loss:.4f}\n"
                )
                f.write(
                    f"mAP: {metrics_result['map']:.4f}, Precision: {metrics_result['map_50']:.4f}, Recall: {metrics_result['mar_100']:.4f}\n"
                )
                for i, ap in enumerate(metrics_result["map_per_class"]):
                    f.write(f"Class {i} AP: {ap:.4f}\n")
                f.write("\n")
    return val_loss, cls_loss, loc_loss, total_loss, metrics_result


def train(train_dataset, val_dataset, args, batch_size, epochs):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Khai báo trực tiếp số lượng proposals
    num_proposals = args.num_proposals  # Sửa số này nếu muốn thay đổi số proposals
    model_ft = model_return(
        type("Args", (), {**vars(args), "num_proposals": num_proposals})()
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01
    )
    scheduler = None

    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft.cuda()

    history = {"train_loss": [], "val_loss": []}

    patience = 7
    delta = 0.1
    early_stopping = EarlyStopping(args, patience=patience, verbose=True, delta=delta)

    is_detection = hasattr(model_ft, "roi_heads")

    # Lấy đường dẫn log
    log_path = getattr(args, "log_path", "training.log")
    # Ghi header log
    with open(log_path, "a", encoding="utf-8") as flog:
        flog.write("\n==== Training Start ====" + "\n")

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
            train_one_epoch(
                model_ft, train_loader, criterion, optimizer, scheduler, epoch
            )
        )
        val_loss, val_cls_loss, val_loc_loss, val_total_loss, val_metrics = (
            val_one_epoch(model_ft, val_loader, criterion, epoch)
        )

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        print(
            f"Epoch: {epoch}/{epochs} \t Avg Train Loss: {train_loss:.3f} \t Avg Valid Loss: {val_loss:.3f}"
        )
        if train_metrics is not None:
            print(
                f"Train mAP: {train_metrics['map']:.4f}, Precision: {train_metrics['map_50']:.4f}, Recall: {train_metrics['mar_100']:.4f}"
            )
            for i, ap in enumerate(train_metrics["map_per_class"]):
                print(f"Train Class {i} AP: {ap:.4f}")
        if val_metrics is not None:
            print(
                f"Val mAP: {val_metrics['map']:.4f}, Precision: {val_metrics['map_50']:.4f}, Recall: {val_metrics['mar_100']:.4f}"
            )
            for i, ap in enumerate(val_metrics["map_per_class"]):
                print(f"Val Class {i} AP: {ap:.4f}")

        # --- Ghi log sau mỗi epoch ---
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(f"Epoch: {epoch}/{epochs}\n")
            flog.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
            flog.write(f"Avg Train Loss: {train_loss:.3f}\n")
            flog.write(f"Avg Valid Loss: {val_loss:.3f}\n")
            if train_metrics is not None:
                flog.write(
                    f"Train mAP: {train_metrics['map']:.4f}, Precision: {train_metrics['map_50']:.4f}, Recall: {train_metrics['mar_100']:.4f}\n"
                )
                for i, ap in enumerate(train_metrics["map_per_class"]):
                    flog.write(f"Train Class {i} AP: {ap:.4f}\n")
            if val_metrics is not None:
                flog.write(
                    f"Val mAP: {val_metrics['map']:.4f}, Precision: {val_metrics['map_50']:.4f}, Recall: {val_metrics['mar_100']:.4f}\n"
                )
                for i, ap in enumerate(val_metrics["map_per_class"]):
                    flog.write(f"Val Class {i} AP: {ap:.4f}\n")
            flog.write("\n")
        # --- End log ---

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        early_stopping(val_loss, model_ft, args, 1, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write("Early stopping\n")
            break

    print()
    print(f"Performance of training:")
    print(
        f"Avg Train Loss: {np.mean(history['train_loss']):.3f} \t Avg Valid Loss: {np.mean(history['val_loss']):.3f}"
    )
    with open(log_path, "a", encoding="utf-8") as flog:
        flog.write(
            f"\nPerformance of training:\nAvg Train Loss: {np.mean(history['train_loss']):.3f} \t Avg Valid Loss: {np.mean(history['val_loss']):.3f}\n"
        )


def get_config(use_argparse=True):
    if use_argparse:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--model_type", dest="model_type", default="resnet_101"
        )
        parser.add_argument(
            "-i", "--image_size", type=int, default=224, dest="image_size"
        )
        parser.add_argument(
            "--log_path", type=str, default="training.log", dest="log_path"
        )
        parser.add_argument(
            "--num_proposals", type=int, default=3, dest="num_proposals"
        )
        args = parser.parse_args()
        return args
    else:

        class Config:
            pass

        config = Config()
        config.model_type = "resnet_101"
        config.image_size = 224
        config.log_path = "training.log"
        config.num_proposals = 3
        return config


if __name__ == "__main__":
    # Đổi use_argparse=True nếu muốn dùng dòng lệnh
    config = get_config(use_argparse=False)

    image_size_tuple = (config.image_size, config.image_size)
    print(f"Model Type : {config.model_type}")
    print(f"Image Size : {image_size_tuple}")

    csv_path = "./dataset/dataset.csv"
    if not os.path.exists(csv_path):
        df = make_dataset_dataframe(
            "dataset/images", "dataset/labels", out_csv=csv_path
        )
    train_csv = pd.read_csv(csv_path)
    train_transform = A.Compose(
        [
            A.Resize(
                config.image_size, config.image_size, interpolation=cv2.INTER_CUBIC, p=1
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
                config.image_size, config.image_size, interpolation=cv2.INTER_CUBIC, p=1
            ),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.01
        ),
    )
    train_dataset = ImageDataset(train_csv, transforms=train_transform)
    val_dataset = ImageDataset(train_csv, transforms=val_transform)

    batch_size = BATCH_SIZE
    epochs = EPOCHS

    torch.manual_seed(42)

    train(
        train_dataset,
        val_dataset,
        config,
        batch_size,
        epochs,
    )
