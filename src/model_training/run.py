"""
Single standalone training run (as opposed to a W&B sweep — see
sweeps/train_sweep.py for that). Logs to both MLflow (Postgres backend,
shared with mlops_project's MLflow server) and W&B.

Usage:
    uv run python -m src.model_training.run --epochs 8 --backbone mobilenet_v2
"""

import argparse
import sys

import wandb

# MLflow prints a run-summary banner containing an emoji; Windows' default
# console codepage (cp1252) can't encode it and the process crashes on exit.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.model_training.train import train_one_run

WANDB_PROJECT = "cv-logistics-bin-count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--backbone", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--unfreeze-layers", type=int, default=0,
        help="0 = fully frozen backbone, -1 = fully unfrozen, N>0 = unfreeze last N blocks",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "manifest": args.manifest,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "unfreeze_layers": args.unfreeze_layers,
        "dropout": args.dropout,
        "run_name": args.run_name,
    }

    wandb.init(project=WANDB_PROJECT, config=config, name=args.run_name)
    result = train_one_run(config)
    wandb.finish()

    print(f"Finished. best_val_loss={result['best_val_loss']:.4f}")
