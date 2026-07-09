"""
Downloads the ~10,441-image counting subset of the Amazon Bin Image Dataset (ABID).

The full ABID (public S3 bucket aft-vbi-pds) is 500k+ images / 50+ GB. This
script instead pulls the curated subset assembled by Udacity for the
"Inventory Monitoring at Distribution Centers" capstone
(https://github.com/udacity/nd009t-capstone-starter/blob/master/starter/file_list.json),
which maps bin item-count (1-5) to ~10k specific image ids from the same bucket.

Images land in <out-dir>/<count>/<image_id>.jpg, e.g. data/raw/3/04031.jpg —
this ImageFolder-style layout is what src/clean_data/run.py consumes next.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "aft-vbi-pds"
FILE_LIST_URL = (
    "https://raw.githubusercontent.com/udacity/nd009t-capstone-starter/"
    "master/starter/file_list.json"
)


def load_file_list(cache_path: Path) -> dict:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FILE_LIST_URL) as resp:
            cache_path.write_bytes(resp.read())
    return json.loads(cache_path.read_text())


def download_subset(out_dir: Path, limit_per_class: int | None = None) -> None:
    file_list = load_file_list(out_dir / "file_list.json")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    total = 0
    for count, paths in file_list.items():
        if limit_per_class:
            paths = paths[:limit_per_class]
        class_dir = out_dir / count
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {len(paths)} images with count={count}")
        for i, meta_path in enumerate(paths, 1):
            image_name = os.path.basename(meta_path).split(".")[0] + ".jpg"
            dest = class_dir / image_name
            if not dest.exists():
                s3.download_file(BUCKET, f"bin-images/{image_name}", str(dest))
            if i % 200 == 0:
                print(f"  [{i}/{len(paths)}]")
            total += 1

    print(f"Done. {total} images under {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument(
        "--limit-per-class", type=int, default=None,
        help="cap images per count class (e.g. 20 for a quick smoke run)",
    )
    args = parser.parse_args()
    download_subset(Path(args.out_dir), args.limit_per_class)
