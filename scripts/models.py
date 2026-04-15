"""Model builders for histology tile classification."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models


BackboneName = Literal["resnet18", "resnet50"]


def build_model(
    backbone: BackboneName,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Build a torchvision ResNet classifier."""
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
