"""
Backbone Networks for Indoor Localization

This module provides various neural network backbones for deep learning-based
indoor localization. All backbones follow a unified interface through BaseBackbone.

Built-in backbones:
    - MLPBackbone: Multi-Layer Perceptron for WiFi fingerprints
    - CNN1DBackbone: 1D CNN for sequential signal processing
    - TimmBackbone: 700+ pretrained models via timm library

Supported architectures (via TimmBackbone):
    - CNN: ResNet, VGG, MobileNet, EfficientNet, ConvNeXt, RegNet, DenseNet
    - Transformer: ViT, Swin, DeiT, BEiT
    - Hybrid: CoAtNet, MaxViT, EfficientFormer

Example:
    >>> import indoorloc as iloc
    >>>
    >>> # Create MLP backbone
    >>> backbone = iloc.MLPBackbone(hidden_dims=[512, 256, 128])
    >>>
    >>> # Create CNN1D backbone
    >>> backbone = iloc.CNN1DBackbone(channels=[64, 128, 256])
    >>>
    >>> # Create ResNet18 backbone (via timm)
    >>> backbone = iloc.TimmBackbone(model_name='resnet18', pretrained=True)
"""
from .base import BaseBackbone, InputAdapter
from .mlp import MLPBackbone
from .cnn1d import CNN1DBackbone
from .timm_wrapper import TimmBackbone

__all__ = [
    'BaseBackbone',
    'InputAdapter',
    'MLPBackbone',
    'CNN1DBackbone',
    'TimmBackbone',
]
