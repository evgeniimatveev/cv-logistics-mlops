"""Backbone factory for the bin-count classifier."""

import torch.nn as nn
from torchvision import models

from src.utils.dataset import NUM_CLASSES


def build_model(backbone: str, freeze_backbone: bool, dropout: float) -> nn.Module:
    if backbone == "mobilenet_v2":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES)
        )
        backbone_params = net.features.parameters()
    elif backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES)
        )
        backbone_params = (
            p for name, p in net.named_parameters() if not name.startswith("fc")
        )
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    if freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False

    return net
