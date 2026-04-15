"""Loss functions for binary and multiclass histology tasks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        probs = torch.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_name: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Create the configured criterion."""
    normalized = loss_name.lower()
    if normalized == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if normalized == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f"Unsupported loss: {loss_name}")
