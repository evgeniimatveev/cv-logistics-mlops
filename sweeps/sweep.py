"""Launches a W&B Bayesian sweep using sweeps/sweep_config.yaml (10 runs)."""

import os

import wandb
import yaml

from train_sweep import train

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WANDB_DIR = os.path.join(REPO_ROOT, "tracking", "wandb")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = WANDB_DIR

SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sweep_config.yaml")

with open(SWEEP_CONFIG_PATH) as f:
    sweep_config = yaml.safe_load(f)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="cv-logistics-bin-count")
    wandb.agent(sweep_id, function=train, count=10)
