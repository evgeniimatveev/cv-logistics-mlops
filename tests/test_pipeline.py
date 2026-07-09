"""
CI-safe smoke tests: no network (S3/W&B) and no live MLflow/Postgres server,
since none of those exist on a GitHub Actions runner. Verifies the dataset
and model-construction code paths work end-to-end on synthetic data.
"""

import csv
from pathlib import Path

import torch
from PIL import Image

from src.utils.dataset import BinCountDataset
from src.utils.model import build_model


def make_synthetic_manifest(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    manifest_path = tmp_path / "manifest.csv"

    rows = []
    for count in range(1, 6):
        for split, n in [("train", 3), ("val", 1), ("test", 1)]:
            for i in range(n):
                img_path = images_dir / f"{count}_{split}_{i}.jpg"
                Image.new("RGB", (64, 64), color=(count * 40, 0, 0)).save(img_path)
                rows.append({"image_path": str(img_path), "count": str(count), "split": split})

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "count", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_dataset_and_model_forward_pass(tmp_path):
    manifest = make_synthetic_manifest(tmp_path)

    train_ds = BinCountDataset(str(manifest), "train")
    assert len(train_ds) == 15  # 5 classes x 3 train images

    image, label = train_ds[0]
    assert image.shape == (3, 224, 224)
    assert 0 <= label.item() <= 4

    model = build_model("mobilenet_v2", unfreeze_layers=0, dropout=0.3)
    batch = torch.stack([train_ds[i][0] for i in range(4)])
    logits = model(batch)
    assert logits.shape == (4, 5)
