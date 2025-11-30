"""
Deep Localizer Module

End-to-end deep learning model for indoor localization, combining
a backbone network with prediction head(s).
"""
from typing import Optional, Dict, Any, Union, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ...registry import BACKBONES, HEADS

if TYPE_CHECKING:
    from ...datasets.base import BaseDataset
    from ...signals.base import BaseSignal
    from ...locations.location import Location, LocalizationResult
    from ...evaluation.metrics import EvaluationResults


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

    # =========================================================================
    # Training Interface (matching BaseLocalizer API)
    # =========================================================================

    def fit(
        self,
        data: Union[List['BaseSignal'], 'BaseDataset'],
        locations: Optional[List['Location']] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_data: Optional['BaseDataset'] = None,
        early_stopping: int = 10,
        verbose: bool = True,
        # Device & precision settings (HuggingFace-style)
        device: Optional[str] = None,
        use_cpu: bool = False,
        fp16: bool = False,
        # DataLoader settings
        num_workers: int = 0,
        pin_memory: bool = True,
        # Training control
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        logging_steps: int = 10,
        **kwargs
    ) -> 'DeepLocalizer':
        """
        Train the deep localizer.

        Supports two calling conventions:
            1. fit(dataset) - Pass a BaseDataset directly
            2. fit(signals, locations) - Pass signals and locations separately

        Args:
            data: Either a BaseDataset or list of training signals
            locations: List of corresponding ground truth locations (required if data is signals)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            val_data: Optional validation dataset for early stopping
            early_stopping: Patience for early stopping (0 to disable)
            verbose: Whether to print training progress
            device: Device to train on ('cuda', 'cuda:0', 'cpu', etc.). Auto-detects if None.
            use_cpu: Force CPU training even if GPU is available (HuggingFace-style)
            fp16: Use mixed precision training (faster on modern GPUs)
            num_workers: Number of DataLoader workers
            pin_memory: Pin memory for faster GPU transfer
            gradient_accumulation_steps: Accumulate gradients over N steps (effective batch = batch_size * N)
            max_grad_norm: Gradient clipping (None to disable)
            logging_steps: Print progress every N epochs
            **kwargs: Additional training arguments

        Returns:
            Self for method chaining

        Example:
            >>> model = iloc.create_model('resnet18', dataset=train)
            >>> model.fit(train, epochs=50, device='cuda', fp16=True)
        """
        from ...datasets.base import BaseDataset
        from ...locations.location import Location
        from ...locations.coordinate import Coordinate

        # Extract data from dataset or signals/locations
        if isinstance(data, BaseDataset):
            X, y = data.to_tensors()
        else:
            if locations is None:
                raise ValueError("locations must be provided when passing signals directly")
            X = np.stack([s.to_numpy() for s in data], axis=0)
            y = np.array([
                [loc.coordinate.x, loc.coordinate.y,
                 loc.floor if loc.floor is not None else 0,
                 int(loc.building_id) if loc.building_id is not None else 0]
                for loc in locations
            ], dtype=np.float32)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Device setup (HuggingFace-style: use_cpu takes priority)
        if use_cpu:
            target_device = torch.device('cpu')
        elif device is not None:
            target_device = torch.device(device)
        else:
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Disable pin_memory for CPU training
        if target_device.type == 'cpu':
            pin_memory = False

        # Create data loader with optimized settings
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Validation data
        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data.to_tensors()
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

        # Setup optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss function
        coord_loss_fn = nn.MSELoss()
        floor_loss_fn = nn.CrossEntropyLoss() if hasattr(self.head, 'num_floors') else None

        # Move model to device
        self.to(target_device)
        if verbose:
            print(f"Training on {target_device}")
            if fp16 and target_device.type == 'cuda':
                print("Using mixed precision (fp16)")

        # Setup mixed precision training (HuggingFace/PyTorch style)
        use_amp = fp16 and target_device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        self._is_trained = False

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(target_device)
                batch_y = batch_y.to(target_device)

                # Mixed precision forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self(batch_x)
                        # Calculate loss
                        if isinstance(outputs, dict):
                            loss = coord_loss_fn(outputs['coords'], batch_y[:, :2])
                            if 'floor_logits' in outputs and floor_loss_fn is not None:
                                loss = loss + floor_loss_fn(outputs['floor_logits'], batch_y[:, 2].long())
                        else:
                            loss = coord_loss_fn(outputs, batch_y[:, :2])

                        # Scale loss for gradient accumulation
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                    # Scaled backward pass
                    scaler.scale(loss).backward()

                    # Step optimizer after accumulation
                    if (step + 1) % gradient_accumulation_steps == 0:
                        if max_grad_norm is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = self(batch_x)
                    # Calculate loss
                    if isinstance(outputs, dict):
                        loss = coord_loss_fn(outputs['coords'], batch_y[:, :2])
                        if 'floor_logits' in outputs and floor_loss_fn is not None:
                            loss = loss + floor_loss_fn(outputs['floor_logits'], batch_y[:, 2].long())
                    else:
                        loss = coord_loss_fn(outputs, batch_y[:, :2])

                    # Scale loss for gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

                    loss.backward()

                    # Step optimizer after accumulation
                    if (step + 1) % gradient_accumulation_steps == 0:
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                epoch_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)

            epoch_loss /= len(train_loader)

            # Validation
            val_loss = epoch_loss
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(target_device)
                        batch_y = batch_y.to(target_device)
                        outputs = self(batch_x)
                        if isinstance(outputs, dict):
                            val_loss += coord_loss_fn(outputs['coords'], batch_y[:, :2]).item()
                        else:
                            val_loss += coord_loss_fn(outputs, batch_y[:, :2]).item()
                val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if verbose and (epoch + 1) % logging_steps == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if early_stopping > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

        self._is_trained = True
        return self

    def evaluate(self, dataset: 'BaseDataset') -> 'EvaluationResults':
        """
        Evaluate the deep localizer on a dataset.

        Args:
            dataset: Test dataset

        Returns:
            EvaluationResults with all metrics

        Example:
            >>> results = model.evaluate(test)
            >>> print(results.summary())
            >>> print(f"Mean Error: {results.mean_error:.2f}m")
        """
        from ...evaluation.metrics import EvaluationResults
        from ...locations.location import Location, LocalizationResult
        from ...locations.coordinate import Coordinate

        if not getattr(self, '_is_trained', False):
            raise RuntimeError("Model must be trained before evaluation. Call fit() first.")

        predictions = self.predict_batch(dataset.signals)
        return EvaluationResults.from_predictions(predictions, dataset.locations)

    def predict_single(self, signal: 'BaseSignal') -> 'LocalizationResult':
        """
        Predict location for a single signal.

        Args:
            signal: Input signal

        Returns:
            Localization result with predicted location
        """
        from ...locations.location import Location, LocalizationResult
        from ...locations.coordinate import Coordinate

        self.eval()
        device = self.device

        x = torch.tensor(signal.to_numpy(), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self(x)

        if isinstance(outputs, dict):
            coords = outputs['coords'][0].cpu().numpy()
            floor = None
            floor_confidence = 0.0
            if 'floor_logits' in outputs:
                floor_proba = torch.softmax(outputs['floor_logits'][0], dim=0)
                floor = int(torch.argmax(floor_proba).item())
                floor_confidence = float(floor_proba[floor].item())
        else:
            coords = outputs[0].cpu().numpy()
            floor = None
            floor_confidence = 0.0

        location = Location(
            coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
            floor=floor,
            confidence=1.0,
            floor_confidence=floor_confidence
        )

        return LocalizationResult(location=location)

    def predict_batch(
        self,
        signals: List['BaseSignal']
    ) -> List['LocalizationResult']:
        """
        Predict locations for multiple signals efficiently.

        Args:
            signals: List of input signals

        Returns:
            List of localization results
        """
        from ...locations.location import Location, LocalizationResult
        from ...locations.coordinate import Coordinate

        self.eval()
        device = self.device

        X = np.stack([s.to_numpy() for s in signals], axis=0)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = self(X_tensor)

        results = []
        for i in range(len(signals)):
            if isinstance(outputs, dict):
                coords = outputs['coords'][i].cpu().numpy()
                floor = None
                floor_confidence = 0.0
                if 'floor_logits' in outputs:
                    floor_proba = torch.softmax(outputs['floor_logits'][i], dim=0)
                    floor = int(torch.argmax(floor_proba).item())
                    floor_confidence = float(floor_proba[floor].item())
            else:
                coords = outputs[i].cpu().numpy()
                floor = None
                floor_confidence = 0.0

            location = Location(
                coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
                floor=floor,
                confidence=1.0,
                floor_confidence=floor_confidence
            )
            results.append(LocalizationResult(location=location))

        return results

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return getattr(self, '_is_trained', False)
