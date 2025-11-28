"""
Classification Head Module

Prediction heads for classification tasks in indoor localization,
such as floor detection and building identification.
"""
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead
from ...registry import HEADS


@HEADS.register_module()
class ClassificationHead(BaseHead):
    """Classification head for discrete predictions.

    Predicts class labels such as floor level or building ID.

    Args:
        in_features: Number of input features from the backbone
        num_classes: Number of output classes
        hidden_dims: Hidden layer dimensions. If None, uses direct projection.
        activation: Activation function ('relu', 'gelu', 'silu')
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        label_smoothing: Label smoothing factor for training (0 = no smoothing)

    Example:
        >>> # Floor classification (4 floors)
        >>> head = ClassificationHead(in_features=512, num_classes=4)
        >>> features = torch.randn(32, 512)
        >>> logits = head(features)  # (32, 4)
        >>> probs = head.predict_proba(features)  # (32, 4)
        >>> floors = head.predict(features)  # (32,)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__(in_features=in_features, dropout=dropout)

        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or []
        self.label_smoothing = label_smoothing

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

        # Final projection to classes
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Class logits, shape (batch, num_classes)
        """
        return self.mlp(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Class probabilities, shape (batch, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Class labels, shape (batch,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    @property
    def output_names(self) -> Tuple[str, ...]:
        return ('logits',)


@HEADS.register_module()
class FloorHead(ClassificationHead):
    """Specialized head for floor classification.

    Convenience class for floor-level prediction with floor-specific
    functionality.

    Args:
        in_features: Number of input features from the backbone
        num_floors: Number of floors to classify
        floor_offset: Starting floor number (e.g., 0, 1, or -1 for basement)
        **kwargs: Additional arguments passed to ClassificationHead

    Example:
        >>> head = FloorHead(in_features=512, num_floors=5, floor_offset=0)
        >>> features = torch.randn(32, 512)
        >>> floors = head.predict_floor(features)  # Floor numbers (0-4)
    """

    def __init__(
        self,
        in_features: int,
        num_floors: int,
        floor_offset: int = 0,
        **kwargs
    ):
        super().__init__(in_features=in_features, num_classes=num_floors, **kwargs)
        self.num_floors = num_floors
        self.floor_offset = floor_offset

    def predict_floor(self, x: torch.Tensor) -> torch.Tensor:
        """Predict floor numbers.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Floor numbers, shape (batch,)
        """
        class_idx = self.predict(x)
        return class_idx + self.floor_offset

    @property
    def output_names(self) -> Tuple[str, ...]:
        return ('floor_logits',)


@HEADS.register_module()
class BuildingHead(ClassificationHead):
    """Specialized head for building classification.

    Convenience class for building-level prediction.

    Args:
        in_features: Number of input features from the backbone
        num_buildings: Number of buildings to classify
        building_names: Optional list of building names for mapping
        **kwargs: Additional arguments passed to ClassificationHead

    Example:
        >>> head = BuildingHead(in_features=512, num_buildings=3)
        >>> features = torch.randn(32, 512)
        >>> buildings = head.predict(features)  # Building indices (0, 1, 2)
    """

    def __init__(
        self,
        in_features: int,
        num_buildings: int,
        building_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(in_features=in_features, num_classes=num_buildings, **kwargs)
        self.num_buildings = num_buildings
        self.building_names = building_names

    def predict_building(self, x: torch.Tensor) -> torch.Tensor:
        """Predict building indices.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Building indices, shape (batch,)
        """
        return self.predict(x)

    def predict_building_name(self, x: torch.Tensor) -> List[str]:
        """Predict building names (if names are provided).

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            List of building names
        """
        if self.building_names is None:
            raise ValueError("Building names not provided")

        indices = self.predict(x).cpu().numpy()
        return [self.building_names[i] for i in indices]

    @property
    def output_names(self) -> Tuple[str, ...]:
        return ('building_logits',)
