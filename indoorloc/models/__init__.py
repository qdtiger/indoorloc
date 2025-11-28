"""
Deep Learning Models for Indoor Localization

This module provides neural network architectures for deep learning-based
indoor localization, including backbone networks, prediction heads, and
end-to-end localizer models.

Components:
    - Backbones: Feature extractors (ResNet, EfficientNet, ViT, Swin, etc.)
    - Heads: Prediction heads (regression, classification, hybrid)
    - Localizers: End-to-end models combining backbone + head

Example:
    >>> import indoorloc as iloc
    >>>
    >>> # Create a deep learning localizer
    >>> model = iloc.DeepLocalizer(
    ...     backbone=dict(
    ...         type='TimmBackbone',
    ...         model_name='resnet18',
    ...         pretrained=True,
    ...         input_type='1d',
    ...     ),
    ...     head=dict(
    ...         type='RegressionHead',
    ...         num_coords=2,
    ...     ),
    ... )
    >>>
    >>> # Or use components directly
    >>> backbone = iloc.TimmBackbone(model_name='efficientnet_b0')
    >>> head = iloc.RegressionHead(in_features=backbone.out_features, num_coords=2)
"""

# Backbones
from .backbones import (
    BaseBackbone,
    InputAdapter,
    TimmBackbone,
)

# Heads
from .heads import (
    BaseHead,
    RegressionHead,
    MultiScaleRegressionHead,
    ClassificationHead,
    FloorHead,
    BuildingHead,
    HybridHead,
    HierarchicalHead,
)

# Localizers
from .localizers import (
    DeepLocalizer,
)

__all__ = [
    # Backbones
    'BaseBackbone',
    'InputAdapter',
    'TimmBackbone',
    # Heads
    'BaseHead',
    'RegressionHead',
    'MultiScaleRegressionHead',
    'ClassificationHead',
    'FloorHead',
    'BuildingHead',
    'HybridHead',
    'HierarchicalHead',
    # Localizers
    'DeepLocalizer',
]
