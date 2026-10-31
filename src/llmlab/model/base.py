"""Общий интерфейс языковых моделей."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class BaseLM(nn.Module):
    """Контракт: forward(idx[B,T], targets=None) -> (logits[B,T,V], loss|None)."""

    arch_name = "base"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        raise NotImplementedError

    def num_params(self) -> int:
        """Число уникальных параметров (общие веса при weight tying считаются один раз)."""
        return sum(p.numel() for p in self.parameters())
