import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import IMG_SIZE, BATCH_SIZE
from dataset import ImageDataset
from model import model_return
from torch import nn, optim

# Dummy dataset, bạn cần thay bằng dataset thực tế
images_dir = "dataset/images"
labels_dir = "dataset/labels"
dataset = ImageDataset(images_dir=images_dir, labels_dir=labels_dir, return_torchvision=True)

batch_size = BATCH_SIZE
shuffle = False
sampler = None
num_workers = 1

def collate_fn(batch):
    # Cho detection, trả về tuple
    if isinstance(batch[0]["target"], dict):
        return {"image": torch.stack([item["image"] for item in batch]),
                "target": {"boxes": [item["target"]["boxes"] for item in batch],
                            "labels": [item["target"]["labels"] for item in batch]}}
    else:
        return {"image": torch.stack([item["image"] for item in batch]),
                "target": torch.stack([item["target"] for item in batch])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", dest="model_type", action="store", default="resnet_101"
    )
    parser.add_argument(
        "-i",
        "--image_size",
        type=int,
        default=IMG_SIZE,
        dest="image_size",
        action="store",
    )
    args = parser.parse_args()

    image_size_tuple = (args.image_size, args.image_size)

    print(f"Model Type : {args.model_type}")
    print(f"Image Size : {image_size_tuple}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = model_return(args)
    model.cuda()
    model.eval()

    # Nếu là classification
    if not hasattr(model, "roi_heads"):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) + nn.L1Loss()
    else:
        criterion = None  # Không cần cho detection

    with torch.no_grad():
        for batch in tqdm(dataloader):
            image = batch["image"].cuda()
            target = batch["target"]
            if hasattr(model, "roi_heads"):  # Detection
                targets = [
                    {"boxes": b.cuda(), "labels": l.cuda()}
                    for b, l in zip(target["boxes"], target["labels"])
                ]
                loss_dict = model(image, targets)
                if isinstance(loss_dict, dict):
                    loss = sum(loss for loss in loss_dict.values())
                    print(f"Detection batch loss: {loss.item():.4f}")
                else:
                    print("Model output is not a dict (no loss computed). Output:", loss_dict)
            else:  # Classification
                labels = target.cuda()
                output = model(image)
                loss = criterion(output, labels)
                print(f"Classification batch loss: {loss.item():.4f}")
