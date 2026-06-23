"""Metrics: callables ``metric(outputs, targets) -> float`` (see src/engine.py)."""

from __future__ import annotations

import torch


class Accuracy:
    """Top-1 classification accuracy for logits of shape (B, num_classes)."""

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        return (preds == targets).float().mean().item()
