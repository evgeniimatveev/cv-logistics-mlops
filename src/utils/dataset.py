"""Manifest-driven PyTorch Dataset for the ABID bin-count task."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 5  # bin item counts 1-5


def build_transforms(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class BinCountDataset(Dataset):
    def __init__(self, manifest_path: str, split: str):
        self.rows = []
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(row)
        self.transform = build_transforms(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["count"]) - 1  # 0-indexed for CrossEntropyLoss
        return image, torch.tensor(label, dtype=torch.long)
