"""L3 deep learning: neural-network fingerprinting on torch (extra ``[deep]``; timm backbones optional).

``DeepLocalizer`` (``"mlp"``, ``"cnn1d"`` or any timm backbone), ``MLPLocalizer`` and
``CNN1DLocalizer`` follow the BaseLocalizer contract: ``fit(X, y, floor=..., building=...,
eval_set=[(X_val, y_val)])``, ``localize``, ``save``/``load_model`` without pickle.

This package is the declared torch bridge of L3. Importing it (or creating a model) does not import
torch; fitting, predicting or loading does. The torch building blocks are importable directly for
custom networks: ``MLPBackbone``, ``CNN1DBackbone``, ``TimmBackbone`` (``backbones``),
``MultiTaskHead``, ``LocalizationNet`` (``heads``) and ``train_model`` (``training``).
"""
from __future__ import annotations

import importlib

from ...core import requires
from .localizer import CNN1DLocalizer, DeepLocalizer, MLPLocalizer

_TORCH_SIDE = {"MLPBackbone": "backbones", "CNN1DBackbone": "backbones", "TimmBackbone": "backbones",
               "build_backbone": "backbones", "MultiTaskHead": "heads", "LocalizationNet": "heads",
               "build_network": "training", "train_model": "training"}

# The torch-side names stay out of __all__: ``from indoorloc.methods.deep import *`` must not need torch.
__all__ = ["CNN1DLocalizer", "DeepLocalizer", "MLPLocalizer"]


def __getattr__(name: str):
    if name in _TORCH_SIDE:  # the torch modules load on first use only
        requires("torch", "deep")  # a missing torch names the extra
        return getattr(importlib.import_module(f"{__name__}.{_TORCH_SIDE[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_TORCH_SIDE])
