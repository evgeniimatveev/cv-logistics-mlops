"""
Minimal FastAPI inference endpoint for the trained bin-count classifier.
Loads straight from the MLflow Model Registry -- whatever version the
"champion" alias points to -- rather than a hardcoded local checkpoint
path, so promoting a new best model (scripts/promote_best_model.py)
takes effect on next request without touching this file.

    uv run uvicorn src.model_deployment.app:app --port 8000
    curl -F "file=@some_bin.jpg" http://localhost:8000/predict
"""

import io

import mlflow
import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.utils.dataset import build_transforms

MODEL_NAME = "cv_logistics_bin_count"
MODEL_ALIAS = "champion"  # falls back to the latest version if unset

with open("config/mlflow_config.yaml") as f:
    _mlflow_cfg = yaml.safe_load(f)["mlflow"]
mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])

app = FastAPI(title="cv-logistics-mlops bin-count inference")
_model = None
_transform = build_transforms(train=False)


def get_model():
    global _model
    if _model is None:
        client = mlflow.tracking.MlflowClient()
        try:
            version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        except mlflow.exceptions.MlflowException:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            version = max(versions, key=lambda v: int(v.version))
        _model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version.version}")
        _model.eval()
    return _model


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = _transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = get_model()(tensor)
        predicted_count = int(logits.argmax(dim=1).item()) + 1  # back to 1-5

    return {"predicted_item_count": predicted_count}
