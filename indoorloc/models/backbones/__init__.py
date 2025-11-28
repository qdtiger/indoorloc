"""
Backbone Networks for Indoor Localization

This module provides various neural network backbones for deep learning-based
indoor localization. All backbones follow a unified interface through BaseBackbone.

Supported architectures (via timm):
    - CNN: ResNet, VGG, MobileNet, EfficientNet, ConvNeXt, RegNet, DenseNet
    - Transformer: ViT, Swin, DeiT, BEiT
    - Hybrid: CoAtNet, MaxViT, EfficientFormer

Example:
    >>> import indoorloc as iloc
    >>>
    >>> # Create ResNet18 backbone
    >>> backbone = iloc.TimmBackbone(model_name='resnet18', pretrained=True)
    >>>
    >>> # List available models
    >>> models = iloc.TimmBackbone.list_aliases()
"""
from .base import BaseBackbone, InputAdapter
from .timm_wrapper import TimmBackbone

__all__ = [
    'BaseBackbone',
    'InputAdapter',
    'TimmBackbone',
]
