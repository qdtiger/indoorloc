"""
Prediction Heads for Indoor Localization

This module provides various prediction heads for deep learning-based
indoor localization tasks.

Head types:
    - RegressionHead: Coordinate prediction (x, y, z)
    - ClassificationHead: Discrete prediction (floor, building)
    - HybridHead: Combined coordinate and classification
    - HierarchicalHead: Coarse-to-fine hierarchical prediction

Example:
    >>> import indoorloc as iloc
    >>>
    >>> # Coordinate regression
    >>> head = iloc.RegressionHead(in_features=512, num_coords=2)
    >>>
    >>> # Floor classification
    >>> head = iloc.FloorHead(in_features=512, num_floors=4)
    >>>
    >>> # Joint coordinate + floor prediction
    >>> head = iloc.HybridHead(in_features=512, num_coords=2, num_floors=4)
"""
from .base import BaseHead
from .regression import RegressionHead, MultiScaleRegressionHead
from .classification import ClassificationHead, FloorHead, BuildingHead
from .hybrid import HybridHead, HierarchicalHead

__all__ = [
    'BaseHead',
    'RegressionHead',
    'MultiScaleRegressionHead',
    'ClassificationHead',
    'FloorHead',
    'BuildingHead',
    'HybridHead',
    'HierarchicalHead',
]
