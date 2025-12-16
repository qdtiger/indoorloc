"""
MLP Backbone for Indoor Localization

Multi-Layer Perceptron backbone optimized for 1D WiFi fingerprint signals.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from ...registry import BACKBONES
from .base import BaseBackbone


def _get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(inplace=True),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name.lower()]


@BACKBONES.register_module()
class MLPBackbone(BaseBackbone):
    """Multi-Layer Perceptron backbone for 1D WiFi fingerprints.

    A simple but effective architecture for WiFi-based indoor localization.
    Uses LazyLinear for automatic input dimension detection.

    Args:
        hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128]).
        activation: Activation function name ('relu', 'gelu', 'leaky_relu', etc.).
        dropout: Dropout rate (0.0-1.0).
        batch_norm: Whether to use batch normalization.
        **kwargs: Additional arguments passed to BaseBackbone.

    Example:
        >>> backbone = MLPBackbone(hidden_dims=[512, 256, 128], dropout=0.3)
        >>> x = torch.randn(32, 520)  # 520 WiFi APs
        >>> features = backbone(x)  # (32, 128)
    """

    def __init__(
        self,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = 'relu',
        dropout: float = 0.3,
        batch_norm: bool = True,
        **kwargs
    ):
        # Force 1D input type for MLP
        kwargs['input_type'] = '1d'
        super().__init__(**kwargs)

        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm

        # Build layers
        layers = []

        # First layer: LazyLinear auto-detects input dimension
        layers.append(nn.LazyLinear(hidden_dims[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self._out_features = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP backbone.

        Args:
            x: Input tensor of shape (batch, features) or (batch, channels, features).

        Returns:
            Feature tensor of shape (batch, out_features).
        """
        # Flatten input: (B, C, F) or (B, F) -> (B, -1)
        x = x.view(x.size(0), -1)
        return self.layers(x)

    @property
    def out_features(self) -> int:
        """Number of output features."""
        return self._out_features

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dims={self.hidden_dims}, "
            f"activation={self.activation_name}, "
            f"dropout={self.dropout_rate}, "
            f"batch_norm={self.use_batch_norm}, "
            f"out_features={self._out_features})"
        )
