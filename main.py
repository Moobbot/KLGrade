from torch.utils.data import DataLoader
from dataset import ImageDataset

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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    for batch in loader:
        images = batch["image"]
        targets = batch["target"]
        # import IPython; IPython.embed()
        # %exit_raise