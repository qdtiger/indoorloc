"""
Base Head Module

Abstract base class for all prediction heads used in indoor localization.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    """Abstract base class for all prediction heads.

    Heads take features from a backbone and produce task-specific outputs
    such as coordinate predictions or floor/building classifications.

    Args:
        in_features: Number of input features from the backbone
        dropout: Dropout rate before final prediction layer

    Example:
        >>> class MyHead(BaseHead):
        ...     def __init__(self, in_features, num_outputs, **kwargs):
        ...         super().__init__(in_features, **kwargs)
        ...         self.fc = nn.Linear(in_features, num_outputs)
        ...
        ...     def forward(self, x):
        ...         return self.fc(x)
        ...
        ...     @property
        ...     def output_names(self):
        ...         return ['output']
    """

    def __init__(
        self,
        in_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.dropout_rate = dropout

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head.

        Args:
            x: Feature tensor from backbone, shape (batch, in_features)

        Returns:
            Prediction tensor(s)
        """
        pass

    @property
    @abstractmethod
    def output_names(self) -> Tuple[str, ...]:
        """Names of the outputs produced by this head."""
        pass

    def get_output_shape(self, in_features: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
        """Get output shapes for each output.

        Args:
            in_features: Number of input features (uses self.in_features if None)

        Returns:
            Dictionary mapping output names to their shapes (without batch dim)
        """
        in_feat = in_features or self.in_features
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        x = torch.zeros(1, in_feat, device=device)

        with torch.no_grad():
            out = self(x)

        if isinstance(out, dict):
            return {k: tuple(v.shape[1:]) for k, v in out.items()}
        else:
            return {self.output_names[0]: tuple(out.shape[1:])}
