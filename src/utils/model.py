"""Backbone factory for the bin-count classifier."""

import torch.nn as nn
from torchvision import models

from src.utils.dataset import NUM_CLASSES


def build_model(backbone: str, unfreeze_layers: int, dropout: float) -> nn.Module:
    """unfreeze_layers: 0 = fully frozen backbone (linear probe),
    -1 = fully unfrozen (full fine-tune), N>0 = unfreeze only the last
    N backbone blocks, freezing everything before them."""
    if backbone == "mobilenet_v2":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES)
        )
        backbone_blocks = list(net.features.children())  # 19 inverted-residual blocks
    elif backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES)
        )
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        backbone_blocks = [stem, net.layer1, net.layer2, net.layer3, net.layer4]
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    if unfreeze_layers == -1:
        return net  # full fine-tune, nothing frozen

    trainable = set(backbone_blocks[len(backbone_blocks) - unfreeze_layers:]) if unfreeze_layers > 0 else set()
    for block in backbone_blocks:
        grad = block in trainable
        for p in block.parameters():
            p.requires_grad = grad

    return net
