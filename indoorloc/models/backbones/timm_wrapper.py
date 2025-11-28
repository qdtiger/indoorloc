"""
Timm Backbone Wrapper

Unified wrapper for all timm models, providing easy access to 700+ pretrained
architectures including ResNet, EfficientNet, ViT, Swin Transformer, etc.

Reference:
    timm (PyTorch Image Models): https://github.com/huggingface/pytorch-image-models
    License: Apache 2.0
"""
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from .base import BaseBackbone, InputAdapter
from ...registry import BACKBONES


def _check_timm_available():
    """Check if timm is installed."""
    try:
        import timm
        return True
    except ImportError:
        return False


@BACKBONES.register_module()
class TimmBackbone(BaseBackbone):
    """Unified wrapper for timm (PyTorch Image Models) backbones.

    Provides access to 700+ pretrained model architectures through a simple
    interface. Automatically handles input channel adaptation and feature
    extraction.

    Supported model families include:
        - CNN: ResNet, VGG, MobileNet, EfficientNet, ConvNeXt, RegNet, etc.
        - Transformer: ViT, Swin, DeiT, BEiT, EVA, etc.
        - Hybrid: CoAtNet, MaxViT, EfficientFormer, etc.

    Args:
        model_name: Name of the timm model (e.g., 'resnet18', 'efficientnet_b0')
        pretrained: Whether to load pretrained ImageNet weights
        in_channels: Number of input channels (will adapt first conv/patch embed)
        input_type: Type of input signal ('1d' for RSSI, '2d' for CSI)
        input_size: Input size. For 1D: num_features. For 2D: (H, W).
        features_only: If True, returns feature maps instead of classification logits
        out_indices: Which feature levels to output (for features_only=True)
        global_pool: Global pooling type ('avg', 'max', 'avgmax', '', None)
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
        target_size: Target size for input adaptation (default: model's default)
        reshape_mode: How to reshape 1D input ('tile', 'reshape', 'conv1d')

    Example:
        >>> import indoorloc as iloc
        >>>
        >>> # Basic usage with ResNet18
        >>> backbone = iloc.TimmBackbone(model_name='resnet18', pretrained=True)
        >>>
        >>> # EfficientNet with custom input channels
        >>> backbone = iloc.TimmBackbone(
        ...     model_name='efficientnet_b0',
        ...     in_channels=1,
        ...     input_type='2d',
        ... )
        >>>
        >>> # Vision Transformer for 1D signals
        >>> backbone = iloc.TimmBackbone(
        ...     model_name='vit_small_patch16_224',
        ...     input_type='1d',
        ...     pretrained=True,
        ... )
        >>>
        >>> # Feature pyramid output
        >>> backbone = iloc.TimmBackbone(
        ...     model_name='resnet50',
        ...     features_only=True,
        ...     out_indices=(1, 2, 3, 4),
        ... )
    """

    # Common model aliases for convenience
    ALIASES = {
        # ResNet family
        'resnet18': 'resnet18',
        'resnet34': 'resnet34',
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        'resnet152': 'resnet152',
        # Wide ResNet
        'wide_resnet50': 'wide_resnet50_2',
        'wide_resnet101': 'wide_resnet101_2',
        # ResNeXt
        'resnext50': 'resnext50_32x4d',
        'resnext101': 'resnext101_32x8d',
        # SE-ResNet
        'seresnet50': 'seresnet50',
        'seresnet101': 'seresnet101',
        # VGG family
        'vgg11': 'vgg11',
        'vgg13': 'vgg13',
        'vgg16': 'vgg16',
        'vgg19': 'vgg19',
        'vgg11_bn': 'vgg11_bn',
        'vgg16_bn': 'vgg16_bn',
        'vgg19_bn': 'vgg19_bn',
        # MobileNet family
        'mobilenetv2': 'mobilenetv2_100',
        'mobilenetv3_small': 'mobilenetv3_small_100',
        'mobilenetv3_large': 'mobilenetv3_large_100',
        # EfficientNet family
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'efficientnet_b2': 'efficientnet_b2',
        'efficientnet_b3': 'efficientnet_b3',
        'efficientnet_b4': 'efficientnet_b4',
        'efficientnet_b5': 'efficientnet_b5',
        'efficientnet_b6': 'efficientnet_b6',
        'efficientnet_b7': 'efficientnet_b7',
        'efficientnetv2_s': 'efficientnetv2_s',
        'efficientnetv2_m': 'efficientnetv2_m',
        'efficientnetv2_l': 'efficientnetv2_l',
        # ConvNeXt family
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'convnext_base': 'convnext_base',
        'convnext_large': 'convnext_large',
        # RegNet family
        'regnetx_002': 'regnetx_002',
        'regnetx_004': 'regnetx_004',
        'regnetx_008': 'regnetx_008',
        'regnety_002': 'regnety_002',
        'regnety_004': 'regnety_004',
        'regnety_008': 'regnety_008',
        # DenseNet family
        'densenet121': 'densenet121',
        'densenet161': 'densenet161',
        'densenet169': 'densenet169',
        'densenet201': 'densenet201',
        # Inception family
        'inception_v3': 'inception_v3',
        'inception_v4': 'inception_v4',
        # Vision Transformer family
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'vit_base': 'vit_base_patch16_224',
        'vit_large': 'vit_large_patch16_224',
        # DeiT family
        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base': 'deit_base_patch16_224',
        # Swin Transformer family
        'swin_tiny': 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base': 'swin_base_patch4_window7_224',
        'swin_large': 'swin_large_patch4_window7_224',
        # SwinV2
        'swinv2_tiny': 'swinv2_tiny_window8_256',
        'swinv2_small': 'swinv2_small_window8_256',
        'swinv2_base': 'swinv2_base_window8_256',
    }

    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        in_channels: int = 1,
        input_type: str = '1d',
        input_size: Optional[Union[int, Tuple[int, int]]] = None,
        features_only: bool = False,
        out_indices: Optional[Tuple[int, ...]] = None,
        global_pool: Optional[str] = 'avg',
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        target_size: Optional[Tuple[int, int]] = None,
        reshape_mode: str = 'tile',
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            input_type=input_type,
            input_size=input_size,
            pretrained=pretrained,
        )

        if not _check_timm_available():
            raise ImportError(
                "timm is required for TimmBackbone.\n"
                "Install with:\n"
                "  pip install timm>=0.9.0 torchvision\n"
                "\n"
                "If you encounter dependency conflicts, install separately:\n"
                "  pip install --no-deps timm\n"
                "  pip install torchvision  # match your torch version"
            )

        import timm

        # Resolve model alias
        self.model_name = self.ALIASES.get(model_name, model_name)
        self.features_only = features_only
        self.out_indices = out_indices or (4,) if features_only else None

        # Determine target size from model config or use default
        if target_size is None:
            # Try to get model's default input size
            try:
                model_cfg = timm.get_pretrained_cfg(self.model_name)
                target_size = model_cfg.input_size[-2:]  # (H, W)
            except Exception:
                target_size = (224, 224)

        self.target_size = target_size

        # Create input adapter for signal preprocessing
        self.input_adapter = InputAdapter(
            input_type=input_type,
            in_channels=in_channels,
            target_size=target_size,
            reshape_mode=reshape_mode,
        )

        # Create timm model
        # For input adapter, we output 3 channels (or in_channels for tile mode)
        model_in_channels = 3 if reshape_mode == 'conv1d' else in_channels

        if features_only:
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                in_chans=model_in_channels,
                features_only=True,
                out_indices=self.out_indices,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                **kwargs
            )
            # Get output channels for each feature level
            self._out_channels = self.model.feature_info.channels()
            self._out_features = self._out_channels[-1]
        else:
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                in_chans=model_in_channels,
                num_classes=0,  # Remove classification head
                global_pool=global_pool,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                **kwargs
            )
            self._out_features = self.model.num_features
            self._out_channels = [self._out_features]

        # Adapt first layer if needed (for non-3-channel input with tile mode)
        if reshape_mode != 'conv1d' and in_channels != 3:
            self._adapt_input_conv(model_in_channels)

    def _adapt_input_conv(self, in_channels: int):
        """Adapt the first convolution layer for different input channels."""
        import timm

        # Try to get the first conv layer
        first_conv_name, first_conv = None, None

        # Common first conv layer names in different architectures
        possible_names = [
            'conv1', 'conv_stem', 'stem.conv', 'patch_embed.proj',
            'features.0', 'stem.0', 'conv1.conv'
        ]

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv_name = name
                first_conv = module
                break

        if first_conv is None:
            return

        # If channels already match, no need to adapt
        if first_conv.in_channels == in_channels:
            return

        # Create new conv layer with adapted input channels
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
        )

        # Initialize weights
        if self.pretrained and first_conv.in_channels == 3:
            # Average the pretrained weights across input channels
            with torch.no_grad():
                if in_channels == 1:
                    # For single channel, average RGB weights
                    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    # For other channels, tile and average
                    weight = first_conv.weight.data
                    new_weight = weight.repeat(1, (in_channels + 2) // 3, 1, 1)[:, :in_channels, :, :]
                    new_conv.weight.data = new_weight

                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data

        # Replace the conv layer
        parts = first_conv_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the backbone.

        Args:
            x: Input tensor
                - For 1D input: (batch, features) or (batch, channels, features)
                - For 2D input: (batch, channels, height, width)

        Returns:
            If features_only=False: Feature tensor of shape (batch, out_features)
            If features_only=True: List of feature maps at different scales
        """
        # Adapt input signal to backbone format
        x = self.input_adapter(x)

        # Forward through timm model
        features = self.model(x)

        # For features_only, return list of feature maps
        if self.features_only:
            return features

        # For single output, ensure it's flattened
        if features.dim() > 2:
            features = features.flatten(1)

        return features

    @property
    def out_features(self) -> int:
        """Number of output features from the backbone."""
        return self._out_features

    @property
    def out_channels_list(self) -> List[int]:
        """Output channels at each feature level (for features_only mode)."""
        return self._out_channels

    def freeze_stages(self, num_stages: int):
        """Freeze first N stages of the backbone.

        Args:
            num_stages: Number of stages to freeze (0 = unfreeze all)
        """
        if num_stages <= 0:
            self.unfreeze()
            return

        # For ResNet-like models
        if hasattr(self.model, 'conv1'):
            if num_stages >= 1:
                for param in self.model.conv1.parameters():
                    param.requires_grad = False
                if hasattr(self.model, 'bn1'):
                    for param in self.model.bn1.parameters():
                        param.requires_grad = False

            stage_names = ['layer1', 'layer2', 'layer3', 'layer4']
            for i, name in enumerate(stage_names[:num_stages - 1]):
                if hasattr(self.model, name):
                    for param in getattr(self.model, name).parameters():
                        param.requires_grad = False

    @classmethod
    def list_models(cls, filter: str = '') -> List[str]:
        """List available timm models.

        Args:
            filter: Filter string to match model names

        Returns:
            List of available model names
        """
        if not _check_timm_available():
            return list(cls.ALIASES.keys())

        import timm
        return timm.list_models(filter, pretrained=True)

    @classmethod
    def list_aliases(cls) -> List[str]:
        """List available model aliases."""
        return list(cls.ALIASES.keys())
