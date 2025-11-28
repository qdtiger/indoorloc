"""
Regression Head Module

Prediction head for coordinate regression in indoor localization.
"""
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from .base import BaseHead
from ...registry import HEADS


@HEADS.register_module()
class RegressionHead(BaseHead):
    """Regression head for coordinate prediction.

    Predicts continuous coordinates (x, y) or (x, y, z) for indoor positioning.

    Args:
        in_features: Number of input features from the backbone
        num_coords: Number of coordinate dimensions (2 for 2D, 3 for 3D)
        hidden_dims: Hidden layer dimensions. If None, uses direct projection.
        activation: Activation function ('relu', 'gelu', 'silu')
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization in hidden layers

    Example:
        >>> head = RegressionHead(in_features=512, num_coords=2)
        >>> features = torch.randn(32, 512)
        >>> coords = head(features)  # (32, 2)

        >>> # With hidden layers
        >>> head = RegressionHead(
        ...     in_features=512,
        ...     num_coords=2,
        ...     hidden_dims=[256, 128],
        ...     dropout=0.1,
        ... )
    """

    def __init__(
        self,
        in_features: int,
        num_coords: int = 2,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__(in_features=in_features, dropout=dropout)

        self.num_coords = num_coords
        self.hidden_dims = hidden_dims or []

        # Activation function
        act_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'leaky_relu': nn.LeakyReLU,
        }.get(activation.lower(), nn.ReLU)

        # Build MLP layers
        layers = []
        prev_dim = in_features

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        # Final projection to coordinates
        layers.append(nn.Linear(prev_dim, num_coords))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict coordinates from features.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Coordinate predictions, shape (batch, num_coords)
        """
        return self.mlp(x)

    @property
    def output_names(self) -> Tuple[str, ...]:
        return ('coords',)


@HEADS.register_module()
class MultiScaleRegressionHead(BaseHead):
    """Multi-scale regression head for hierarchical coordinate prediction.

    Uses multiple prediction scales for coarse-to-fine localization.

    Args:
        in_features: Number of input features from the backbone
        num_coords: Number of coordinate dimensions
        num_scales: Number of prediction scales
        hidden_dim: Hidden dimension for each scale
        dropout: Dropout rate

    Example:
        >>> head = MultiScaleRegressionHead(in_features=512, num_coords=2, num_scales=3)
        >>> features = torch.randn(32, 512)
        >>> coords = head(features)  # Final prediction (32, 2)
        >>> coarse, medium, fine = head.get_all_scales(features)
    """

    def __init__(
        self,
        in_features: int,
        num_coords: int = 2,
        num_scales: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(in_features=in_features, dropout=dropout)

        self.num_coords = num_coords
        self.num_scales = num_scales

        # Scale predictors (coarse to fine)
        self.scale_heads = nn.ModuleList()
        for i in range(num_scales):
            self.scale_heads.append(nn.Sequential(
                nn.Linear(in_features if i == 0 else in_features + num_coords, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_coords),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict coordinates using multi-scale refinement.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Final coordinate predictions, shape (batch, num_coords)
        """
        # Coarse prediction
        pred = self.scale_heads[0](x)

        # Refine through subsequent scales
        for head in self.scale_heads[1:]:
            combined = torch.cat([x, pred], dim=1)
            residual = head(combined)
            pred = pred + residual

        return pred

    def get_all_scales(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions at all scales.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            List of predictions at each scale
        """
        predictions = []
        pred = self.scale_heads[0](x)
        predictions.append(pred)

        for head in self.scale_heads[1:]:
            combined = torch.cat([x, pred], dim=1)
            residual = head(combined)
            pred = pred + residual
            predictions.append(pred)

        return predictions

    @property
    def output_names(self) -> Tuple[str, ...]:
        return ('coords',)
