import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dataset import ImageDataset
from early_stop import EarlyStopping
from model import model_return
from config import BATCH_SIZE, EPOCHS, K_FOLDS


def train_for_kfold(model, dataloader, criterion, optimizer, scheduler, fold, epoch):
    train_loss = 0.0
    model.train()  # Model을 Train Mode로 변환 >> Dropout Layer 같은 경우 Train시 동작 해야 함
    with torch.set_grad_enabled(True):
        for batch in tqdm(
            dataloader, desc=f"Fold {fold} Epoch {epoch} Train", unit="Batch"
        ):
            optimizer.zero_grad()
            image = batch["image"].cuda()
            target = batch["target"]

            # Nếu là detection (FasterRCNN)
            if hasattr(model, "roi_heads"):
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                # loss_dict gồm: loss_classifier, loss_box_reg, ...
                loss = sum(loss for loss in loss_dict.values())
                train_loss += loss.item()
            else:
                # Classification như cũ
                labels = batch["target"].cuda()
                output = model(image)
                loss = criterion(output, labels)
                train_loss += loss.item()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return train_loss


def val_for_kfold(model, dataloader, criterion, fold, epoch):
    val_loss = 0.0
    model.eval()  # Model을 Eval Mode로 전환 >> Dropout Layer 같은 경우 Eval시 동작 하지 않아야 함
    with torch.no_grad():  # gradient 연산 기록 추적 off
        for batch in tqdm(
            dataloader, desc=f"Fold {fold} Epoch {epoch} Valid", unit="Batch"
        ):
            image = batch["image"].cuda()
            target = batch["target"]
            # Nếu là detection (FasterRCNN)
            if hasattr(model, "roi_heads"):
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
            else:
                labels = batch["target"].cuda()
                output = model(image)
                loss = criterion(output, labels)
                val_loss += loss.item()
    return val_loss


def train(
    train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf
):
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(np.arange(len(train_dataset)), labels), start=1
    ):
        # Data Load에 사용되는 index, key의 순서를 지정하는데 사용, Sequential , Random, SubsetRandom, Batch 등 + Sampler
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # Data Load
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

        model_ft = model_return(args)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Loss Function
        # criterion = nn.MSELoss()
        # criterion = my_ce_mse_loss

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01
        )  # Optimizer
        scheduler = None

        if torch.cuda.device_count() > 1:
            model_ft = nn.DataParallel(
                model_ft
            )  # model이 여러 대의 gpu에 할당되도록 병렬 처리
        model_ft.cuda()  # Model을 GPU에 할당

        history = {"train_loss": [], "val_loss": []}

        patience = 7
        delta = 0.1
        early_stopping = EarlyStopping(
            args, patience=patience, verbose=True, delta=delta
        )

        for epoch in range(1, epochs + 1):
            if epoch == 2:
                for param in model_ft.parameters():
                    param.requires_grad = True

                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model_ft.parameters()),
                    weight_decay=0.0001,
                    lr=0.001,
                )
                # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
                scheduler = MultiStepLR(optimizer, milestones=[2], gamma=0.1)

            print(f"Learning Rate : {optimizer.param_groups[0]['lr']}")

            train_loss = train_for_kfold(
                model_ft, train_loader, criterion, optimizer, scheduler, fold, epoch
            )
            val_loss = val_for_kfold(model_ft, val_loader, criterion, fold, epoch)

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            print(
                f"Epoch: {epoch}/{epochs} \t Avg Train Loss: {train_loss:.3f} \t Avg Valid Loss: {val_loss:.3f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            early_stopping(val_loss, model_ft, args, fold, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        foldperf[f"fold{fold}"] = history

    tl_f, vall_f = [], []

    for f in range(1, k + 1):
        tl_f.append(np.mean(foldperf[f"fold{f}"]["train_loss"]))
        vall_f.append(np.mean(foldperf[f"fold{f}"]["val_loss"]))

    print()
    print(f"Performance of {k} Fold Cross Validation")
    print(
        f"Avg Train Loss: {np.mean(tl_f):.3f} \t Avg Valid Loss: {np.mean(vall_f):.3f}"
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

    train_csv = pd.read_csv("./KneeXray/train/train.csv")

    train_transform = A.Compose(
        [
            A.Resize(
                args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1
            ),
            # A.RandomCrop(height=int(384*0.8), width=int(384*0.8), p=1),
            # A.GridDistortion(p=0.5),
            # A.ElasticTransform(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=1),
            A.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # -1 ~ 1의 범위를 가지도록 정규화
            ToTensorV2(),  # 0 ~ 1의 범위를 가지도록 정규화
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(
                args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1
            ),
            A.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # -1 ~ 1의 범위를 가지도록 정규화
            ToTensorV2(),  # 0 ~ 1의 범위를 가지도록 정규화
        ]
    )
    train_dataset = ImageDataset(
        train_csv, transforms=train_transform, return_torchvision=True
    )
    val_dataset = ImageDataset(
        train_csv, transforms=val_transform, return_torchvision=True
    )

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    k = K_FOLDS
    torch.manual_seed(42)
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = train_dataset.get_labels()
    foldperf = {}

    train(
        train_dataset,
        val_dataset,
        args,
        batch_size,
        epochs,
        k,
        splits,
        labels,
        foldperf,
    )
