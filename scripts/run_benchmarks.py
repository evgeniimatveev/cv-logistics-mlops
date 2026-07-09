"""
Runs a fixed comparison sweep (frozen -> partial -> full fine-tune)
sequentially, each as its own MLflow + W&B run, so results land side
by side and can be compared with scripts/generate_benchmarks_md.py.

Learning rate drops as more of the backbone unfreezes -- fine-tuning
pretrained ImageNet weights with the same lr used for a fresh linear
head destroys them (catastrophic forgetting), so each config gets a
smaller step size the more of the network is trainable.

Usage:
    uv run python scripts/run_benchmarks.py
"""

import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.model_training.train import train_one_run  # noqa: E402

WANDB_PROJECT = "cv-logistics-bin-count"

BASE = {
    "manifest": "data/processed/manifest.csv",
    "backbone": "mobilenet_v2",
    "batch_size": 32,
    "dropout": 0.3,
    "epochs": 5,
}

CONFIGS = [
    {"run_name": "bench_frozen", "unfreeze_layers": 0, "learning_rate": 0.001},
    {"run_name": "bench_partial2", "unfreeze_layers": 2, "learning_rate": 0.0003},
    {"run_name": "bench_partial4", "unfreeze_layers": 4, "learning_rate": 0.0001},
    {"run_name": "bench_full", "unfreeze_layers": -1, "learning_rate": 0.00005},
]

if __name__ == "__main__":
    for overrides in CONFIGS:
        config = {**BASE, **overrides}
        print(f"=== BENCH START {config['run_name']} ===")

        wandb.init(
            project=WANDB_PROJECT, config=config, name=config["run_name"], reinit=True
        )
        result = train_one_run(config)
        wandb.finish()

        print(f"=== BENCH DONE {config['run_name']} best_val_loss={result['best_val_loss']:.4f} ===")

    print("=== ALL BENCHMARKS DONE ===")
