"""
Deep Learning Localizers

This module provides end-to-end deep learning models for indoor localization
that combine backbone networks with prediction heads.

Example:
    >>> import indoorloc as iloc
    >>>
    >>> # Create localizer with ResNet backbone
    >>> model = iloc.DeepLocalizer(
    ...     backbone=dict(type='TimmBackbone', model_name='resnet18'),
    ...     head=dict(type='RegressionHead', num_coords=2),
    ... )
"""
from .deep_localizer import DeepLocalizer

__all__ = [
    'DeepLocalizer',
]
