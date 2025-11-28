"""
Deep Localizer Module

End-to-end deep learning model for indoor localization, combining
a backbone network with prediction head(s).
"""
from typing import Optional, Dict, Any, Union, List, Tuple

import torch
import torch.nn as nn

from ...registry import BACKBONES, HEADS


class DeepLocalizer(nn.Module):
    """End-to-end deep learning model for indoor localization.

    Combines a backbone network (feature extractor) with one or more
    prediction heads for coordinate regression and/or classification.

    Args:
        backbone: Backbone configuration dict or nn.Module instance.
            Dict format: {'type': 'TimmBackbone', 'model_name': 'resnet18', ...}
        head: Head configuration dict or nn.Module instance.
            Dict format: {'type': 'RegressionHead', 'num_coords': 2, ...}
        neck: Optional neck module between backbone and head
        freeze_backbone: Whether to freeze backbone weights
        init_cfg: Initialization configuration

    Example:
        >>> import indoorloc as iloc
        >>>
        >>> # Method 1: Using config dicts
        >>> model = iloc.DeepLocalizer(
        ...     backbone=dict(
        ...         type='TimmBackbone',
        ...         model_name='resnet18',
        ...         pretrained=True,
        ...         input_type='1d',
        ...     ),
        ...     head=dict(
        ...         type='RegressionHead',
        ...         num_coords=2,
        ...         hidden_dims=[256, 128],
        ...     ),
        ... )
        >>>
        >>> # Method 2: Using pre-built modules
        >>> backbone = iloc.TimmBackbone(model_name='resnet18')
        >>> head = iloc.RegressionHead(in_features=backbone.out_features, num_coords=2)
        >>> model = iloc.DeepLocalizer(backbone=backbone, head=head)
        >>>
        >>> # Forward pass
        >>> signal = torch.randn(32, 520)  # 520 WiFi APs
        >>> coords = model(signal)  # (32, 2)
        >>>
        >>> # With hybrid head
        >>> model = iloc.DeepLocalizer(
        ...     backbone=dict(type='TimmBackbone', model_name='efficientnet_b0'),
        ...     head=dict(type='HybridHead', num_coords=2, num_floors=4),
        ... )
        >>> outputs = model(signal)
        >>> # outputs['coords']: (32, 2)
        >>> # outputs['floor_logits']: (32, 4)
    """

    def __init__(
        self,
        backbone: Union[Dict[str, Any], nn.Module],
        head: Union[Dict[str, Any], nn.Module],
        neck: Optional[nn.Module] = None,
        freeze_backbone: bool = False,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__()

        # Build backbone
        if isinstance(backbone, dict):
            self.backbone = BACKBONES.build(backbone)
        else:
            self.backbone = backbone

        # Get backbone output features
        backbone_out_features = self.backbone.out_features

        # Optional neck
        self.neck = neck
        if neck is not None:
            neck_out_features = neck.out_features
        else:
            neck_out_features = backbone_out_features

        # Build head
        if isinstance(head, dict):
            # Auto-fill in_features if not provided
            if 'in_features' not in head:
                head = dict(head)  # Make a copy
                head['in_features'] = neck_out_features
            self.head = HEADS.build(head)
        else:
            self.head = head

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        # Initialize weights
        if init_cfg is not None:
            self.init_weights(init_cfg)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        **head_kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple]:
        """Forward pass through the localizer.

        Args:
            x: Input signal tensor
                - For 1D (RSSI): (batch, num_aps) or (batch, channels, num_aps)
                - For 2D (CSI): (batch, channels, height, width)
            return_features: If True, also return backbone features
            **head_kwargs: Additional arguments passed to head

        Returns:
            If head returns dict: Dict of predictions
            If head returns tensor: Prediction tensor
            If return_features=True: Tuple of (predictions, features)
        """
        # Extract features
        features = self.backbone(x)

        # Apply neck if present
        if self.neck is not None:
            features = self.neck(features)

        # Apply head
        predictions = self.head(features, **head_kwargs)

        if return_features:
            return predictions, features
        return predictions

    def predict(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Get final predictions (inference mode).

        Convenience method that ensures model is in eval mode and
        returns clean predictions.

        Args:
            x: Input signal tensor
            **kwargs: Additional arguments

        Returns:
            Dictionary of predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, **kwargs)

            # Wrap single tensor in dict
            if isinstance(outputs, torch.Tensor):
                return {'coords': outputs}
            return outputs

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        if hasattr(self.backbone, 'freeze'):
            self.backbone.freeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        if hasattr(self.backbone, 'unfreeze'):
            self.backbone.unfreeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def freeze_backbone_stages(self, num_stages: int):
        """Freeze first N stages of backbone.

        Args:
            num_stages: Number of stages to freeze
        """
        if hasattr(self.backbone, 'freeze_stages'):
            self.backbone.freeze_stages(num_stages)

    def get_parameter_groups(
        self,
        backbone_lr_mult: float = 0.1,
        head_lr_mult: float = 1.0,
    ) -> List[Dict]:
        """Get parameter groups with different learning rates.

        Useful for fine-tuning where backbone should have lower LR.

        Args:
            backbone_lr_mult: Learning rate multiplier for backbone
            head_lr_mult: Learning rate multiplier for head

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = [
            {
                'params': self.backbone.parameters(),
                'lr_mult': backbone_lr_mult,
                'name': 'backbone',
            },
            {
                'params': self.head.parameters(),
                'lr_mult': head_lr_mult,
                'name': 'head',
            },
        ]

        if self.neck is not None:
            groups.append({
                'params': self.neck.parameters(),
                'lr_mult': head_lr_mult,
                'name': 'neck',
            })

        return groups

    def init_weights(self, init_cfg: Dict):
        """Initialize model weights.

        Args:
            init_cfg: Initialization configuration
                - type: 'kaiming', 'xavier', 'normal', 'constant'
                - layers: Which layers to initialize ('Linear', 'Conv2d', etc.)
        """
        init_type = init_cfg.get('type', 'kaiming')
        layers = init_cfg.get('layers', ['Linear'])

        for module in self.modules():
            if any(isinstance(module, getattr(nn, layer, type(None))) for layer in layers):
                if init_type == 'kaiming':
                    if hasattr(module, 'weight') and module.weight is not None:
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    if hasattr(module, 'weight') and module.weight is not None:
                        nn.init.xavier_uniform_(module.weight)
                elif init_type == 'normal':
                    if hasattr(module, 'weight') and module.weight is not None:
                        std = init_cfg.get('std', 0.01)
                        nn.init.normal_(module.weight, mean=0, std=std)
                elif init_type == 'constant':
                    if hasattr(module, 'weight') and module.weight is not None:
                        val = init_cfg.get('val', 0)
                        nn.init.constant_(module.weight, val)

                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @property
    def device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device

    def to_onnx(
        self,
        filepath: str,
        input_shape: Tuple[int, ...],
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None,
    ):
        """Export model to ONNX format.

        Args:
            filepath: Output ONNX file path
            input_shape: Input tensor shape (without batch dim)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
        """
        self.eval()
        dummy_input = torch.randn(1, *input_shape, device=self.device)

        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        print(f"Model exported to {filepath}")

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """Get model summary.

        Args:
            input_shape: Optional input shape for computing output shapes

        Returns:
            Model summary string
        """
        lines = [
            "DeepLocalizer Summary",
            "=" * 50,
            f"Backbone: {self.backbone.__class__.__name__}",
            f"  - Output features: {self.backbone.out_features}",
        ]

        if hasattr(self.backbone, 'model_name'):
            lines.append(f"  - Model: {self.backbone.model_name}")

        if self.neck is not None:
            lines.append(f"Neck: {self.neck.__class__.__name__}")

        lines.extend([
            f"Head: {self.head.__class__.__name__}",
            f"  - Input features: {self.head.in_features}",
            f"  - Outputs: {self.head.output_names}",
        ])

        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.extend([
            "-" * 50,
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
        ])

        if input_shape is not None:
            lines.append("-" * 50)
            lines.append(f"Input shape: {input_shape}")
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, *input_shape, device=self.device)
                    out = self(dummy)
                    if isinstance(out, dict):
                        for k, v in out.items():
                            lines.append(f"Output '{k}': {tuple(v.shape[1:])}")
                    else:
                        lines.append(f"Output shape: {tuple(out.shape[1:])}")
            except Exception as e:
                lines.append(f"Could not compute output shape: {e}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
