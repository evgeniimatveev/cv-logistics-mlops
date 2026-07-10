"""
Core training loop for the bin item-count classifier.

Shared by src/model_training/run.py (single standalone run) and
sweeps/train_sweep.py (invoked by a W&B sweep agent) so both paths log
to the same MLflow experiment / Postgres backend and the same W&B project.
"""

import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from src.utils.dataset import BinCountDataset
from src.utils.model import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_CONFIG_PATH = REPO_ROOT / "config" / "mlflow_config.yaml"


def load_mlflow_config() -> dict:
    with open(MLFLOW_CONFIG_PATH) as f:
        return yaml.safe_load(f)["mlflow"]


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train() if train else model.eval()
    total_loss, correct, abs_err, n = 0.0, 0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            abs_err += (preds - labels).abs().sum().item()
            n += images.size(0)

    return {"loss": total_loss / n, "accuracy": correct / n, "mae": abs_err / n}


def train_one_run(config: dict) -> dict:
    """config keys: backbone, learning_rate, batch_size, epochs,
    unfreeze_layers, dropout, manifest, models_dir, wandb_project, run_name"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BinCountDataset(config["manifest"], "train")
    val_ds = BinCountDataset(config["manifest"], "val")
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    model = build_model(
        config["backbone"], config["unfreeze_layers"], config["dropout"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    mlflow_cfg = load_mlflow_config()
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    best_val_loss = float("inf")
    best_val_mae = None
    models_dir = Path(config.get("models_dir", REPO_ROOT / "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_model.pt"

    with mlflow.start_run(run_name=config.get("run_name")):
        mlflow.log_params(
            {k: v for k, v in config.items() if k not in ("manifest", "models_dir")}
        )

        for epoch in range(1, config["epochs"] + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            log_payload = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_mae": val_metrics["mae"],
            }
            mlflow.log_metrics(log_payload, step=epoch)
            if wandb.run is not None:
                wandb.log(log_payload, step=epoch)

            print(
                f"epoch {epoch}/{config['epochs']} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.3f} "
                f"val_mae={val_metrics['mae']:.3f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_mae = val_metrics["mae"]
                torch.save(model.state_dict(), best_path)

        # log/register the checkpoint that actually had the best val_loss,
        # not whatever the model holds after the final epoch -- those are
        # the same thing only if val_loss never regresses, which isn't
        # true once a run trains long enough to start overfitting
        model.load_state_dict(torch.load(best_path, map_location=device))

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_mae", best_val_mae)
        example_input, _ = next(iter(val_loader))
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input[:1].numpy(),
            serialization_format="pickle",
            registered_model_name="cv_logistics_bin_count",
        )

    return {"best_val_loss": best_val_loss, "best_val_mae": best_val_mae, **val_metrics}
