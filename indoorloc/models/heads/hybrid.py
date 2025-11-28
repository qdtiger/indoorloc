"""
Hybrid Head Module

Combined prediction head for multi-task indoor localization,
handling both coordinate regression and floor/building classification.
"""
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead
from ...registry import HEADS


@HEADS.register_module()
class HybridHead(BaseHead):
    """Hybrid head for joint coordinate and classification prediction.

    Combines coordinate regression with floor and/or building classification
    in a multi-task learning setup. Useful for 3D indoor localization where
    both (x, y) coordinates and floor level need to be predicted.

    Args:
        in_features: Number of input features from the backbone
        num_coords: Number of coordinate dimensions (default: 2 for x, y)
        num_floors: Number of floors (None to disable floor prediction)
        num_buildings: Number of buildings (None to disable building prediction)
        hidden_dim: Hidden dimension for shared and task-specific layers
        shared_layers: Number of shared hidden layers before task branches
        dropout: Dropout rate
        task_weights: Optional dict of task weights for loss computation
            {'coords': 1.0, 'floor': 1.0, 'building': 0.5}

    Example:
        >>> # Joint coordinate and floor prediction
        >>> head = HybridHead(
        ...     in_features=512,
        ...     num_coords=2,
        ...     num_floors=4,
        ...     num_buildings=3,
        ... )
        >>> features = torch.randn(32, 512)
        >>> outputs = head(features)
        >>> # outputs['coords']: (32, 2)
        >>> # outputs['floor_logits']: (32, 4)
        >>> # outputs['building_logits']: (32, 3)
    """

    def __init__(
        self,
        in_features: int,
        num_coords: int = 2,
        num_floors: Optional[int] = None,
        num_buildings: Optional[int] = None,
        hidden_dim: int = 256,
        shared_layers: int = 1,
        dropout: float = 0.0,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(in_features=in_features, dropout=dropout)

        self.num_coords = num_coords
        self.num_floors = num_floors
        self.num_buildings = num_buildings
        self.hidden_dim = hidden_dim
        self.task_weights = task_weights or {
            'coords': 1.0,
            'floor': 1.0,
            'building': 1.0,
        }

        # Shared layers
        shared = []
        prev_dim = in_features
        for _ in range(shared_layers):
            shared.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*shared) if shared else nn.Identity()
        shared_out_dim = hidden_dim if shared_layers > 0 else in_features

        # Coordinate regression branch
        self.coord_head = nn.Sequential(
            nn.Linear(shared_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_coords),
        )

        # Floor classification branch
        if num_floors is not None:
            self.floor_head = nn.Sequential(
                nn.Linear(shared_out_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_floors),
            )
        else:
            self.floor_head = None

        # Building classification branch
        if num_buildings is not None:
            self.building_head = nn.Sequential(
                nn.Linear(shared_out_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_buildings),
            )
        else:
            self.building_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all prediction branches.

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Dictionary containing:
                - 'coords': Coordinate predictions (batch, num_coords)
                - 'floor_logits': Floor logits (batch, num_floors) if enabled
                - 'building_logits': Building logits (batch, num_buildings) if enabled
        """
        # Shared feature extraction
        shared_feat = self.shared(x)

        outputs = {}

        # Coordinate prediction
        outputs['coords'] = self.coord_head(shared_feat)

        # Floor prediction
        if self.floor_head is not None:
            outputs['floor_logits'] = self.floor_head(shared_feat)

        # Building prediction
        if self.building_head is not None:
            outputs['building_logits'] = self.building_head(shared_feat)

        return outputs

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get final predictions (coordinates and class labels).

        Args:
            x: Feature tensor, shape (batch, in_features)

        Returns:
            Dictionary containing:
                - 'coords': Coordinate predictions (batch, num_coords)
                - 'floor': Floor labels (batch,) if enabled
                - 'building': Building labels (batch,) if enabled
        """
        outputs = self.forward(x)
        predictions = {'coords': outputs['coords']}

        if 'floor_logits' in outputs:
            predictions['floor'] = torch.argmax(outputs['floor_logits'], dim=-1)

        if 'building_logits' in outputs:
            predictions['building'] = torch.argmax(outputs['building_logits'], dim=-1)

        return predictions

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        coord_loss_fn=None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            outputs: Output dict from forward()
            targets: Target dict with 'coords', 'floor', 'building'
            coord_loss_fn: Loss function for coordinates (default: MSE)

        Returns:
            Dictionary of losses for each task and total weighted loss
        """
        coord_loss_fn = coord_loss_fn or nn.MSELoss()
        losses = {}

        # Coordinate loss
        if 'coords' in targets:
            losses['coord_loss'] = coord_loss_fn(outputs['coords'], targets['coords'])

        # Floor loss
        if 'floor_logits' in outputs and 'floor' in targets:
            losses['floor_loss'] = F.cross_entropy(
                outputs['floor_logits'], targets['floor']
            )

        # Building loss
        if 'building_logits' in outputs and 'building' in targets:
            losses['building_loss'] = F.cross_entropy(
                outputs['building_logits'], targets['building']
            )

        # Total weighted loss
        total = 0
        if 'coord_loss' in losses:
            total = total + self.task_weights.get('coords', 1.0) * losses['coord_loss']
        if 'floor_loss' in losses:
            total = total + self.task_weights.get('floor', 1.0) * losses['floor_loss']
        if 'building_loss' in losses:
            total = total + self.task_weights.get('building', 1.0) * losses['building_loss']

        losses['total_loss'] = total
        return losses

    @property
    def output_names(self) -> Tuple[str, ...]:
        names = ['coords']
        if self.floor_head is not None:
            names.append('floor_logits')
        if self.building_head is not None:
            names.append('building_logits')
        return tuple(names)


@HEADS.register_module()
class HierarchicalHead(BaseHead):
    """Hierarchical head for coarse-to-fine localization.

    Predicts location in a hierarchical manner: building -> floor -> coordinates.
    Each level conditions on the previous prediction.

    Args:
        in_features: Number of input features from the backbone
        num_buildings: Number of buildings
        num_floors_per_building: Number of floors per building (can be int or list)
        num_coords: Number of coordinate dimensions
        hidden_dim: Hidden dimension
        dropout: Dropout rate

    Example:
        >>> head = HierarchicalHead(
        ...     in_features=512,
        ...     num_buildings=3,
        ...     num_floors_per_building=4,  # Same for all buildings
        ...     num_coords=2,
        ... )
    """

    def __init__(
        self,
        in_features: int,
        num_buildings: int,
        num_floors_per_building: int,
        num_coords: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(in_features=in_features, dropout=dropout)

        self.num_buildings = num_buildings
        self.num_floors = num_floors_per_building
        self.num_coords = num_coords

        # Building classifier
        self.building_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_buildings),
        )

        # Floor classifier (conditioned on building embedding)
        self.building_embed = nn.Embedding(num_buildings, hidden_dim // 4)
        self.floor_head = nn.Sequential(
            nn.Linear(in_features + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_floors_per_building),
        )

        # Coordinate regressor (conditioned on building and floor)
        self.floor_embed = nn.Embedding(num_floors_per_building, hidden_dim // 4)
        self.coord_head = nn.Sequential(
            nn.Linear(in_features + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_coords),
        )

    def forward(
        self,
        x: torch.Tensor,
        building_labels: Optional[torch.Tensor] = None,
        floor_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Hierarchical forward pass.

        During training, use ground truth labels for conditioning.
        During inference, use predicted labels.

        Args:
            x: Feature tensor, shape (batch, in_features)
            building_labels: Optional ground truth building labels
            floor_labels: Optional ground truth floor labels

        Returns:
            Dictionary with all predictions
        """
        batch_size = x.shape[0]

        # Building prediction
        building_logits = self.building_head(x)

        # Get building conditioning
        if building_labels is not None:
            building_idx = building_labels
        else:
            building_idx = torch.argmax(building_logits, dim=-1)

        building_emb = self.building_embed(building_idx)

        # Floor prediction (conditioned on building)
        floor_input = torch.cat([x, building_emb], dim=-1)
        floor_logits = self.floor_head(floor_input)

        # Get floor conditioning
        if floor_labels is not None:
            floor_idx = floor_labels
        else:
            floor_idx = torch.argmax(floor_logits, dim=-1)

        floor_emb = self.floor_embed(floor_idx)

        # Coordinate prediction (conditioned on building and floor)
        coord_input = torch.cat([x, building_emb, floor_emb], dim=-1)
        coords = self.coord_head(coord_input)

        return {
            'building_logits': building_logits,
            'floor_logits': floor_logits,
            'coords': coords,
        }

    @property
    def output_names(self) -> Tuple[str, ...]:
        return ('building_logits', 'floor_logits', 'coords')
