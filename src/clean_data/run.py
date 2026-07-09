"""
Builds a train/val/test manifest from the downloaded ABID subset.

Reads data/raw/<count>/<image_id>.jpg (written by src/download_data/run.py),
drops unreadable/corrupt files, and writes data/processed/manifest.csv with
columns: image_path, count, split — stratified by count so each split keeps
the same class balance. Images are left in place (referenced, not copied)
to avoid doubling disk usage.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_manifest(raw_dir: Path) -> list[dict]:
    rows = []
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        count = class_dir.name
        for img_path in class_dir.glob("*.jpg"):
            if is_valid_image(img_path):
                rows.append({"image_path": str(img_path), "count": count})
            else:
                print(f"skipping corrupt image: {img_path}")
    return rows


def stratified_split(rows: list[dict]) -> list[dict]:
    train, rest = train_test_split(
        rows, train_size=SPLIT_RATIOS["train"],
        stratify=[r["count"] for r in rows], random_state=42,
    )
    val_frac = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
    val, test = train_test_split(
        rest, train_size=val_frac,
        stratify=[r["count"] for r in rest], random_state=42,
    )
    for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
        for r in split_rows:
            r["split"] = split_name
    return train + val + test


def write_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "count", "split"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/manifest.csv")
    args = parser.parse_args()

    rows = build_manifest(Path(args.raw_dir))
    rows = stratified_split(rows)
    write_manifest(rows, Path(args.out))

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in SPLIT_RATIOS}
    print(f"Wrote {len(rows)} rows to {args.out} -- {counts}")
