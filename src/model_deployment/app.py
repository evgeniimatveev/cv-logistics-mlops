"""
Minimal FastAPI inference endpoint for the trained bin-count classifier.

    uv run uvicorn src.model_deployment.app:app --port 8000
    curl -F "file=@some_bin.jpg" http://localhost:8000/predict
"""

import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.utils.dataset import build_transforms
from src.utils.model import build_model

MODEL_PATH = "models/best_model.pt"
BACKBONE = "mobilenet_v2"

app = FastAPI(title="cv-logistics-mlops bin-count inference")
_model = None
_transform = build_transforms(train=False)


def get_model():
    global _model
    if _model is None:
        _model = build_model(BACKBONE, unfreeze_layers=0, dropout=0.3)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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
