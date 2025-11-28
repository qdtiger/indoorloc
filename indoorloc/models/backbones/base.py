"""
Base Backbone Module

Abstract base class for all neural network backbones used in indoor localization.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn


class InputAdapter(nn.Module):
    """Adapts 1D/2D input signals to backbone expected format.

    For indoor localization, inputs can be:
    - 1D: RSSI vectors (num_aps,) or (batch, num_aps)
    - 2D: CSI matrices (subcarriers, antennas) or similar

    This adapter converts them to the format expected by image-based backbones.

    Args:
        input_type: Type of input signal ('1d' or '2d')
        in_channels: Number of input channels
        target_size: Target spatial size for the backbone (H, W)
        reshape_mode: How to reshape 1D input ('tile', 'reshape', 'conv1d')
    """

    def __init__(
        self,
        input_type: str = '1d',
        in_channels: int = 1,
        target_size: Tuple[int, int] = (224, 224),
        reshape_mode: str = 'tile',
    ):
        super().__init__()
        self.input_type = input_type
        self.in_channels = in_channels
        self.target_size = target_size
        self.reshape_mode = reshape_mode

        # For 1D conv mode
        if input_type == '1d' and reshape_mode == 'conv1d':
            # Use 1D convolution to project to 2D feature map
            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            # Project to target channels
            self.proj = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input to backbone expected format.

        Args:
            x: Input tensor
                - 1D: (batch, features) or (batch, channels, features)
                - 2D: (batch, channels, height, width)

        Returns:
            Tensor of shape (batch, channels, H, W) for 2D backbones
        """
        if self.input_type == '1d':
            return self._adapt_1d(x)
        else:
            return self._adapt_2d(x)

    def _adapt_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt 1D signal to 2D format."""
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Add channel dimension if needed: (batch, features) -> (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, channels, num_features = x.shape
        H, W = self.target_size

        if self.reshape_mode == 'tile':
            # Tile the 1D signal to form a 2D image
            # Repeat along height dimension
            x = x.unsqueeze(2).expand(-1, -1, H, -1)  # (B, C, H, features)
            # Interpolate to target width
            x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        elif self.reshape_mode == 'reshape':
            # Reshape 1D to 2D (requires num_features to be factorable)
            # Find closest factorization
            import math
            sqrt_n = int(math.sqrt(num_features))
            h = sqrt_n
            w = (num_features + h - 1) // h

            # Pad if needed
            pad_size = h * w - num_features
            if pad_size > 0:
                x = nn.functional.pad(x, (0, pad_size))

            x = x.view(batch_size, channels, h, w)
            x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        elif self.reshape_mode == 'conv1d':
            # Use learned 1D conv then reshape
            x = self.conv1d(x)  # (B, 64, features)
            # Reshape to 2D
            x = x.unsqueeze(2).expand(-1, -1, H, -1)
            x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = self.proj(x)  # (B, 3, H, W)

        return x

    def _adapt_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt 2D signal to expected size."""
        # Ensure 4D: (batch, channels, height, width)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Resize to target size if needed
        if x.shape[-2:] != self.target_size:
            x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return x


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all backbone networks.

    All backbone implementations should inherit from this class and implement
    the required abstract methods.

    Args:
        in_channels: Number of input channels (default: 1 for single-channel signals)
        input_type: Type of input ('1d' for RSSI vectors, '2d' for CSI matrices)
        input_size: Expected input size. For 1D: number of features. For 2D: (H, W).
        pretrained: Whether to use pretrained weights (if available)

    Example:
        >>> class MyBackbone(BaseBackbone):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.layers = nn.Sequential(...)
        ...         self._out_features = 512
        ...
        ...     def forward(self, x):
        ...         return self.layers(x)
        ...
        ...     @property
        ...     def out_features(self):
        ...         return self._out_features
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_type: str = '1d',
        input_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_type = input_type
        self.input_size = input_size
        self.pretrained = pretrained

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor of shape (batch, out_features) or (batch, C, H, W)
        """
        pass

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Number of output features (channels) from the backbone."""
        pass

    @property
    def out_channels(self) -> int:
        """Alias for out_features for compatibility."""
        return self.out_features

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape given input shape.

        Args:
            input_shape: Input tensor shape (without batch dimension)

        Returns:
            Output tensor shape (without batch dimension)
        """
        device = next(self.parameters()).device
        x = torch.zeros(1, *input_shape, device=device)
        with torch.no_grad():
            out = self(x)
        return tuple(out.shape[1:])

    def freeze(self, freeze_bn: bool = True):
        """Freeze all parameters in the backbone.

        Args:
            freeze_bn: Whether to also freeze batch normalization layers
        """
        for param in self.parameters():
            param.requires_grad = False

        if freeze_bn:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

    def unfreeze(self):
        """Unfreeze all parameters in the backbone."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_stages(self, num_stages: int):
        """Freeze first N stages of the backbone.

        This is a no-op in the base class. Subclasses with stage-based
        architecture should override this method.

        Args:
            num_stages: Number of stages to freeze
        """
        pass
