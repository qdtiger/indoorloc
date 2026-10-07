"""Feature extractors (backbones) for the deep localizers: MLP, 1-D CNN and any timm image model.

Part of the declared torch bridge ``methods/deep`` (extra ``[deep]``): this module imports torch at
module level and is loaded only when a deep model is fitted or loaded. timm is imported only when a
timm backbone is built. Every backbone maps ``(B, *input_shape)`` to ``(B, out_features)``.

Ported from 0.1 ``indoorloc.models.backbones`` with these changes: the input size is explicit
(no ``LazyLinear``/``LazyConv1d``, so a saved ``state_dict`` loads without a dummy forward pass);
the CNN1D pools to ``output_length`` positions instead of one (see ``CNN1DBackbone``); an unknown
pooling type is an error instead of silently becoming average pooling; timm models are
created with ``pretrained=False`` unless asked, receive the input channels through timm's own
``in_chans`` adaptation, and see a 1-D fingerprint folded into a near-square image (the 0.1
``reshape`` mode) instead of the 0.1 default of tiling it into a 224 x 224 image.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ...core import requires

__all__ = ["CNN1DBackbone", "MLPBackbone", "TimmBackbone", "build_backbone"]

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh,
                "leaky_relu": lambda: nn.LeakyReLU(0.1)}


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; available: {sorted(_ACTIVATIONS)}") from None


class MLPBackbone(nn.Module):
    """Fully connected layers ``Linear -> [BatchNorm] -> activation -> [Dropout]`` on flattened input.

    The usual deep model for RSSI fingerprints (e.g. Kim et al., 2018). ``hidden=()`` is the
    identity, which with a linear head gives a linear model (used as a closed-form test).

    References
    ----------
    K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building
    and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018.
    https://doi.org/10.1186/s41044-018-0031-2
    """

    def __init__(self, in_features: int, hidden=(512, 256, 128), activation: str = "relu",
                 dropout: float = 0.3, batch_norm: bool = True):
        super().__init__()
        layers, width = [], int(in_features)
        for h in hidden:
            layers.append(nn.Linear(width, int(h)))
            if batch_norm:
                layers.append(nn.BatchNorm1d(int(h)))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = int(h)
        self.layers = nn.Sequential(*layers)
        self.out_features = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.flatten(1))


class CNN1DBackbone(nn.Module):
    """1-D convolutions along the feature axis, then max (or average) pooling to a fixed length.

    ``Conv1d(kernel k, stride s, padding k // 2) -> [BatchNorm] -> ReLU -> [Dropout]`` per entry of
    ``channels``, then adaptive pooling to ``output_length`` positions, flattened to
    ``channels[-1] * output_length`` features. Input ``(B, F)`` is one channel; ``(B, C, F)`` has C
    channels (e.g. antennas); more leading axes are merged into channels. ``kernel_sizes``/``strides``
    shorter than ``channels`` repeat their last value (as in 0.1).

    ``output_length=1`` is the 0.1 global pooling. It makes the features invariant to *where* along
    the AP axis a pattern occurs, i.e. it discards which AP was heard, the very information of an
    RSSI fingerprint; the default keeps a coarse position axis instead. ``projection=P`` first maps
    the flattened input through ``Linear(F, P) -> [BatchNorm] -> ReLU`` and convolves the P learned
    features (one channel). The AP index order of an RSSI vector carries no geometry, so convolving it
    directly generalizes poorly; CNNLoc (Song et al., 2019) likewise encodes the fingerprint with a
    (stacked auto-)encoder before its 1-D CNN. Leave ``projection=None`` when the convolved axis is
    physical (CSI subcarriers, time).

    References
    ----------
    X. Song, X. Fan, C. Xiang, Q. Ye, L. Liu, Z. Wang, X. He, N. Yang and G. Fang, "A Novel
    Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting",
    IEEE Access 7:110698-110709, 2019. https://doi.org/10.1109/ACCESS.2019.2933921
    """

    def __init__(self, in_channels: int = 1, channels=(64, 128, 256), kernel_sizes=(7, 5, 3),
                 strides=(2, 2, 2), pooling: str = "max", output_length: int = 8, dropout: float = 0.3,
                 batch_norm: bool = True, projection: int | None = None, in_features: int | None = None):
        super().__init__()
        self.project = None
        if projection:  # a learned dense encoder first, so the convolved axis has a learned order
            if in_features is None:
                raise ValueError("projection needs in_features (the flattened input size)")
            layers = [nn.Linear(int(in_features), int(projection))]
            layers += [nn.BatchNorm1d(int(projection))] if batch_norm else []
            self.project = nn.Sequential(*layers, nn.ReLU())
            in_channels = 1
        if pooling not in ("max", "avg"):
            raise ValueError(f"pooling must be 'max' or 'avg', got {pooling!r}")
        if int(output_length) < 1:
            raise ValueError(f"output_length must be >= 1, got {output_length}")
        channels = [int(c) for c in channels]

        def pad(values):  # shorter lists repeat their last value (the 0.1 rule)
            values = [int(x) for x in values]
            return values[:len(channels)] + values[-1:] * max(0, len(channels) - len(values))

        kernel_sizes, strides = pad(kernel_sizes), pad(strides)
        layers, c_in = [], int(in_channels)
        for c, k, s in zip(channels, kernel_sizes, strides):
            layers.append(nn.Conv1d(c_in, c, kernel_size=k, stride=s, padding=k // 2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            c_in = c
        self.in_channels = int(in_channels)
        self.conv = nn.Sequential(*layers)
        pool = nn.AdaptiveMaxPool1d if pooling == "max" else nn.AdaptiveAvgPool1d
        self.pool = pool(int(output_length))
        self.out_features = c_in * int(output_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(len(x), self.in_channels, -1) if self.project is None else self.project(x.flatten(1))[:, None]
        return self.pool(self.conv(x)).flatten(1)


def image_layout(input_shape) -> tuple[int, tuple[int, int], bool]:
    """How a sample of ``input_shape`` becomes an image: ``(channels, (H, W), fold)``.

    ``(F,)`` is folded row by row into a zero-padded ``H x W`` square (``H = ceil(sqrt(F))``);
    ``(H, W)`` is one channel; ``(C, H, W)`` is kept; more leading axes are merged into channels.
    """
    shape = tuple(int(s) for s in input_shape)
    if len(shape) == 1:
        h = math.isqrt(shape[0] - 1) + 1 if shape[0] > 1 else 1
        return 1, (h, -(-shape[0] // h)), True
    if len(shape) == 2:
        return 1, shape, False
    return math.prod(shape[:-2]), shape[-2:], False


class TimmBackbone(nn.Module):
    """Any timm image model (ResNet, EfficientNet, ConvNeXt, ViT, ...) as a fingerprint encoder.

    The model is created with ``num_classes=0`` (pooled features) and ``in_chans`` set from the
    input, so timm adapts the first layer itself. 1-D fingerprints are folded into an image (see
    ``image_layout``); ``image_size=(H, W)`` additionally resizes bilinearly (needed by models with
    a fixed input size such as ViTs, which also take ``img_size`` in ``timm_kwargs``). Extra keyword
    arguments go to ``timm.create_model`` (e.g. ``drop_rate``, ``drop_path_rate``).
    ``pretrained=False`` by default: ImageNet weights are downloaded only when asked for.

    References
    ----------
    R. Wightman, "PyTorch Image Models", GitHub repository, 2019.
    https://doi.org/10.5281/zenodo.4414861
    """

    def __init__(self, model_name: str, input_shape, pretrained: bool = False, image_size=None, **timm_kwargs):
        super().__init__()
        timm = requires("timm", "deep")
        if not timm.is_model(model_name):
            raise ValueError(f"unknown backbone {model_name!r}: use 'mlp', 'cnn1d' or a timm model name "
                             "(see timm.list_models())")
        self.in_chans, self.grid, self.fold = image_layout(input_shape)
        self.image_size = None if image_size is None else tuple(int(s) for s in image_size)
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=self.in_chans,
                                       num_classes=0, **timm_kwargs)
        self.out_features = int(self.model.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fold:
            h, w = self.grid
            x = F.pad(x.reshape(len(x), -1), (0, h * w - x[0].numel())).reshape(len(x), 1, h, w)
        else:
            x = x.reshape(len(x), self.in_chans, *self.grid)
        if self.image_size is not None:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        return self.model(x).flatten(1)


def build_backbone(name: str, input_shape, **kwargs) -> nn.Module:
    """``"mlp"``, ``"cnn1d"`` or a timm model name -> a module with ``out_features``."""
    shape = tuple(int(s) for s in input_shape)
    if name == "mlp":
        return MLPBackbone(math.prod(shape), **kwargs)
    if name == "cnn1d":
        return CNN1DBackbone(math.prod(shape[:-1]) if len(shape) > 1 else 1, in_features=math.prod(shape), **kwargs)
    try:
        requires("timm", "deep")
    except ImportError as err:
        raise ImportError(f"backbone {name!r} is neither 'mlp' nor 'cnn1d', so it names a timm model: "
                          f"{err}") from err
    return TimmBackbone(name, shape, **kwargs)
