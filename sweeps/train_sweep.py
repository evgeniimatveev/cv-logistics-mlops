"""Training function invoked by the W&B sweep agent (see sweeps/sweep.py)."""

import sys
from pathlib import Path

import wandb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_training.train import train_one_run  # noqa: E402

DEFAULTS = {
    "manifest": "data/processed/manifest.csv",
    "backbone": "mobilenet_v2",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 6,
    "unfreeze_layers": 0,
    "dropout": 0.3,
}


def train():
    wandb.init(config=DEFAULTS)
    config = {**DEFAULTS, **dict(wandb.config)}
    train_one_run(config)
    wandb.finish()


if __name__ == "__main__":
    train()
