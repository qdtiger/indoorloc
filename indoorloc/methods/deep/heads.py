"""Prediction heads for the deep localizers and the backbone + head network (torch bridge).

``MultiTaskHead`` is the 0.1 ``HybridHead`` without its shared layer (the backbone already is the
shared part): one branch regresses the (standardized) coordinates, optional branches classify the
floor and the building. ``LocalizationNet`` chains a backbone and a head and returns a dict of
tensors, ``{"pos": (B, D), "floor": (B, n_floors), "building": (B, n_buildings)}``, with the
classification entries present only when the training data had those labels.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["LocalizationNet", "MultiTaskHead"]


def _branch(in_features: int, hidden, out_features: int, dropout: float) -> nn.Module:
    layers, width = [], int(in_features)
    for h in hidden:
        layers += [nn.Linear(width, int(h)), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = int(h)
    layers.append(nn.Linear(width, int(out_features)))
    return nn.Sequential(*layers)


class MultiTaskHead(nn.Module):
    """Coordinate regression plus optional floor and building classification (multi-task learning).

    Each task has its own branch ``[Linear -> ReLU -> Dropout] * len(hidden) -> Linear``; with
    ``hidden=()`` the branches are linear maps of the shared features. ``n_floors``/``n_buildings``
    of 0 disable a task.

    References
    ----------
    R. Caruana, "Multitask Learning", Machine Learning 28:41-75, 1997.
    https://doi.org/10.1023/A:1007379606734
    """

    def __init__(self, in_features: int, n_outputs: int, n_floors: int = 0, n_buildings: int = 0,
                 hidden=(), dropout: float = 0.0):
        super().__init__()
        self.pos = _branch(in_features, hidden, n_outputs, dropout)
        self.floor = _branch(in_features, hidden, n_floors, dropout) if n_floors else None
        self.building = _branch(in_features, hidden, n_buildings, dropout) if n_buildings else None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"pos": self.pos(features)}
        if self.floor is not None:
            out["floor"] = self.floor(features)
        if self.building is not None:
            out["building"] = self.building(features)
        return out


class LocalizationNet(nn.Module):
    """``head(backbone(x))``: the network a DeepLocalizer trains."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(x))
