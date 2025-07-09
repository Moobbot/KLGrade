import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import IMG_SIZE
from dataset import ImageDataset
from torch import nn, optim
from model import model_return

images_dir = "dataset/images"
labels_dir = "dataset/labels"

dataset = ImageDataset(images_dir=images_dir, labels_dir=labels_dir)

batch_size = 4
num_workers = 1
shuffle = False
sampler = None


def collate_fn(batch):
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    import torch

    images = torch.stack(images, dim=0)
    return {"image": images, "target": targets}


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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) + nn.L1Loss()
    train_loss = 0.0
    model_ft = model_return(args)
    model = model_ft

    model.train()  # Model을 Train Mode로 변환 >> Dropout Layer 같은 경우 Train시 동작 해야 함
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        weight_decay=0.0001,
        lr=0.001,
    )
    with torch.set_grad_enabled(
        True
    ):  # with문 : 자원의 효율적 사용, 객체의 life cycle을 설계 가능, 항상(True) gradient 연산 기록을 추적
        for batch in tqdm(dataloader):
            images = batch["image"]
            targets = batch["target"]
            optimizer.zero_grad()  # 반복 시 gradient(기울기)를 0으로 초기화, gradient는 += 되기 때문
            image, labels = (
                batch["image"].cuda(),
                batch["target"].cuda(),
            )  # Tensor를 GPU에 할당

            # labels = F.one_hot(labels, num_classes=5).float() # nn.MSELoss() 사용 시 필요
            output = model(
                image
            )  # image(data)를 model에 넣어서 hypothesis(가설) 값을 획득

            loss = criterion(output, labels)  # Error, Prediction Loss 계산
            train_loss += loss.item()  # loss.item()을 통해 Loss의 스칼라 값을 가져온다.

            loss.backward()  # Prediction Loss를 Back Propagation으로 계산
            optimizer.step()  # optimizer를 이용해 Loss를 효율적으로 최소화 할 수 있게 Parameter 수정

        # import IPython; IPython.embed()
        # %exit_raise
