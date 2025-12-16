"""
CNN1D Backbone for Indoor Localization

1D Convolutional Neural Network backbone for sequential signal processing.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from ...registry import BACKBONES
from .base import BaseBackbone


@BACKBONES.register_module()
class CNN1DBackbone(BaseBackbone):
    """1D Convolutional Neural Network backbone for sequential signals.

    Optimized for 1D WiFi/BLE fingerprint processing without reshaping
    to 2D images.

    Args:
        channels: Channel progression through conv layers (e.g., [64, 128, 256]).
        kernel_sizes: Kernel sizes for each conv layer.
        strides: Stride for each conv layer.
        pooling: Pooling type ('max', 'avg', or None).
        dropout: Dropout rate (0.0-1.0).
        batch_norm: Whether to use batch normalization.
        **kwargs: Additional arguments passed to BaseBackbone.

    Example:
        >>> backbone = CNN1DBackbone(channels=[64, 128, 256], kernel_sizes=[7, 5, 3])
        >>> x = torch.randn(32, 520)  # 520 WiFi APs
        >>> features = backbone(x)  # (32, 256)
    """

    def __init__(
        self,
        channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 3],
        strides: List[int] = [2, 2, 2],
        pooling: Optional[str] = 'max',
        dropout: float = 0.3,
        batch_norm: bool = True,
        **kwargs
    ):
        # Force 1D input type
        kwargs['input_type'] = '1d'
        super().__init__(**kwargs)

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pooling_type = pooling
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm

        # Validate parameter lengths
        n_layers = len(channels)
        if len(kernel_sizes) != n_layers:
            kernel_sizes = kernel_sizes[:n_layers] if len(kernel_sizes) > n_layers else \
                           kernel_sizes + [kernel_sizes[-1]] * (n_layers - len(kernel_sizes))
        if len(strides) != n_layers:
            strides = strides[:n_layers] if len(strides) > n_layers else \
                      strides + [strides[-1]] * (n_layers - len(strides))

        # Build conv layers
        layers = []

        # First conv layer: LazyConv1d auto-detects input channels
        layers.append(nn.LazyConv1d(
            channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=kernel_sizes[0] // 2
        ))
        if batch_norm:
            layers.append(nn.BatchNorm1d(channels[0]))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Subsequent conv layers
        for i in range(n_layers - 1):
            layers.append(nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_sizes[i + 1],
                stride=strides[i + 1],
                padding=kernel_sizes[i + 1] // 2
            ))
            if batch_norm:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)

        # Global pooling for stable output regardless of input size
        if pooling == 'max':
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'avg':
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # Default to avg

        self._out_features = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN1D backbone.

        Args:
            x: Input tensor of shape (batch, features) or (batch, channels, features).

        Returns:
            Feature tensor of shape (batch, out_features).
        """
        # Handle input shape
        if x.dim() == 2:
            # (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            # (features,) -> (1, 1, features)
            x = x.unsqueeze(0).unsqueeze(0)

        # Apply conv layers
        x = self.conv_layers(x)

        # Global pooling
        x = self.global_pool(x)

        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.flatten(1)

        return x

    @property
    def out_features(self) -> int:
        """Number of output features."""
        return self._out_features

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"channels={self.channels}, "
            f"kernel_sizes={self.kernel_sizes}, "
            f"strides={self.strides}, "
            f"pooling={self.pooling_type}, "
            f"dropout={self.dropout_rate}, "
            f"out_features={self._out_features})"
        )
