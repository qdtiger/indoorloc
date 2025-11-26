"""
Hybrid Signal Implementation

Provides a container for multi-modal sensor fusion, combining different
signal types (WiFi, BLE, IMU, Magnetometer, etc.) for hybrid localization.
"""
from typing import Dict, Optional, Any, List
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@SIGNALS.register_module()
class HybridSignal(BaseSignal):
    """Hybrid signal container for multi-modal sensor fusion.

    Combines multiple signal types (WiFi, BLE, IMU, Magnetometer, UWB, etc.)
    into a single unified signal representation for hybrid localization.

    Args:
        sensors: Dictionary mapping signal type names to BaseSignal instances.
            Keys: 'wifi', 'ble', 'imu', 'magnetometer', 'uwb', 'vlc', 'ultrasound'
        fusion_weights: Optional weights for each sensor modality.
        metadata: Signal metadata.

    Example:
        >>> from indoorloc.signals import WiFiSignal, MagnetometerSignal
        >>> import numpy as np
        >>>
        >>> # Create individual signals
        >>> wifi = WiFiSignal(rssi_values=np.random.randn(100))
        >>> mag = MagnetometerSignal(
        ...     magnetic_field=np.random.randn(10, 3) * 10 + 40
        ... )
        >>>
        >>> # Combine into hybrid signal
        >>> hybrid = HybridSignal(sensors={
        ...     'wifi': wifi,
        ...     'magnetometer': mag
        ... })
        >>>
        >>> # Access individual signals
        >>> wifi_signal = hybrid.get_signal('wifi')
        >>> mag_signal = hybrid.get_signal('magnetometer')
        >>>
        >>> # Get combined tensor
        >>> tensor = hybrid.to_tensor()  # Concatenates all signals
    """

    # Supported signal types
    SUPPORTED_SIGNAL_TYPES = [
        'wifi', 'ble', 'imu', 'magnetometer', 'uwb', 'vlc', 'ultrasound'
    ]

    def __init__(
        self,
        sensors: Dict[str, BaseSignal],
        fusion_weights: Optional[Dict[str, float]] = None,
        metadata: Optional[SignalMetadata] = None,
        **kwargs
    ):
        # Validate sensors dictionary
        if not sensors:
            raise ValueError("sensors dictionary cannot be empty")

        for signal_type, signal in sensors.items():
            if not isinstance(signal, BaseSignal):
                raise TypeError(
                    f"Signal '{signal_type}' must be a BaseSignal instance, "
                    f"got {type(signal)}"
                )

        self.sensors = sensors

        # Initialize fusion weights (default: equal weights)
        if fusion_weights is None:
            n = len(sensors)
            self.fusion_weights = {k: 1.0 / n for k in sensors.keys()}
        else:
            # Normalize weights to sum to 1
            total = sum(fusion_weights.values())
            self.fusion_weights = {
                k: v / total for k, v in fusion_weights.items()
            }

        # Pass data to parent class
        super().__init__(self.sensors, metadata)

    @property
    def signal_type(self) -> str:
        """Return the signal type identifier."""
        return 'hybrid'

    @property
    def feature_dim(self) -> int:
        """Return the total feature dimension (sum of all signals)."""
        return sum(signal.feature_dim for signal in self.sensors.values())

    @property
    def signal_types(self) -> List[str]:
        """Return list of signal types in this hybrid signal."""
        return list(self.sensors.keys())

    @property
    def num_modalities(self) -> int:
        """Return the number of sensor modalities."""
        return len(self.sensors)

    def get_signal(self, signal_type: str) -> Optional[BaseSignal]:
        """Get a specific signal by type.

        Args:
            signal_type: Type of signal to retrieve (e.g., 'wifi', 'magnetometer').

        Returns:
            BaseSignal instance, or None if not found.
        """
        return self.sensors.get(signal_type)

    def has_signal(self, signal_type: str) -> bool:
        """Check if a specific signal type is present.

        Args:
            signal_type: Type of signal to check.

        Returns:
            True if signal type is present, False otherwise.
        """
        return signal_type in self.sensors

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert hybrid signal to PyTorch tensor.

        Concatenates all sensor signals into a single tensor.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').

        Returns:
            Concatenated tensor of all signals.
        """
        # Sort by signal type for consistent ordering
        sorted_types = sorted(self.sensors.keys())

        tensors = []
        for signal_type in sorted_types:
            signal = self.sensors[signal_type]
            tensor = signal.to_tensor(device)

            # Flatten to 1D if necessary
            if tensor.dim() > 1:
                tensor = tensor.flatten()

            tensors.append(tensor)

        return torch.cat(tensors, dim=0)

    def to_numpy(self) -> np.ndarray:
        """Convert hybrid signal to NumPy array.

        Concatenates all sensor signals into a single array.

        Returns:
            Concatenated array of all signals.
        """
        # Sort by signal type for consistent ordering
        sorted_types = sorted(self.sensors.keys())

        arrays = []
        for signal_type in sorted_types:
            signal = self.sensors[signal_type]
            array = signal.to_numpy()

            # Flatten to 1D if necessary
            if array.ndim > 1:
                array = array.flatten()

            arrays.append(array)

        return np.concatenate(arrays)

    def to_dict_by_modality(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary with separate arrays for each modality.

        Returns:
            Dictionary mapping signal types to their numpy arrays.
        """
        return {
            signal_type: signal.to_numpy()
            for signal_type, signal in self.sensors.items()
        }

    def to_weighted_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert to weighted tensor using fusion weights.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').

        Returns:
            Weighted concatenated tensor.
        """
        sorted_types = sorted(self.sensors.keys())

        tensors = []
        for signal_type in sorted_types:
            signal = self.sensors[signal_type]
            tensor = signal.to_tensor(device)

            # Flatten to 1D if necessary
            if tensor.dim() > 1:
                tensor = tensor.flatten()

            # Apply fusion weight
            weight = self.fusion_weights[signal_type]
            tensors.append(tensor * weight)

        return torch.cat(tensors, dim=0)

    def normalize(self, method: str = 'standard') -> 'HybridSignal':
        """Normalize all signals independently.

        Args:
            method: Normalization method to apply to each signal.

        Returns:
            New HybridSignal with normalized signals.
        """
        normalized_sensors = {}

        for signal_type, signal in self.sensors.items():
            try:
                normalized_sensors[signal_type] = signal.normalize(method)
            except Exception as e:
                print(f"Warning: Could not normalize {signal_type} signal: {e}")
                normalized_sensors[signal_type] = signal

        return HybridSignal(
            sensors=normalized_sensors,
            fusion_weights=self.fusion_weights,
            metadata=self.metadata
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HybridSignal':
        """Create HybridSignal from dictionary.

        Args:
            d: Dictionary containing signal data.

        Returns:
            HybridSignal instance.

        Note:
            This requires that signal classes are properly registered.
        """
        from ..registry import SIGNALS

        sensors = {}
        sensor_data = d.get('sensors', {})

        for signal_type, signal_dict in sensor_data.items():
            # Get the signal class from registry
            signal_class_name = signal_dict.get('signal_class')
            if signal_class_name:
                SignalClass = SIGNALS.get(signal_class_name)
                sensors[signal_type] = SignalClass.from_dict(signal_dict)
            else:
                print(f"Warning: Could not load {signal_type} signal (no class specified)")

        fusion_weights = d.get('fusion_weights')

        return cls(
            sensors=sensors,
            fusion_weights=fusion_weights
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert HybridSignal to dictionary.

        Returns:
            Dictionary representation of hybrid signal.
        """
        return {
            'signal_type': self.signal_type,
            'sensors': {
                signal_type: {
                    'signal_class': signal.__class__.__name__,
                    **signal.to_dict()
                }
                for signal_type, signal in self.sensors.items()
            },
            'fusion_weights': self.fusion_weights,
            'signal_types': self.signal_types,
            'num_modalities': self.num_modalities,
            'feature_dim': self.feature_dim
        }

    def get_feature_dims_by_modality(self) -> Dict[str, int]:
        """Get feature dimensions for each modality.

        Returns:
            Dictionary mapping signal types to their feature dimensions.
        """
        return {
            signal_type: signal.feature_dim
            for signal_type, signal in self.sensors.items()
        }

    def __repr__(self) -> str:
        """String representation."""
        signal_types_str = ', '.join(self.signal_types)
        return (
            f"HybridSignal(modalities=[{signal_types_str}], "
            f"feature_dim={self.feature_dim})"
        )

    def __len__(self) -> int:
        """Return the number of sensor modalities."""
        return self.num_modalities

    def __contains__(self, signal_type: str) -> bool:
        """Check if signal type is present using 'in' operator."""
        return signal_type in self.sensors

    def __getitem__(self, signal_type: str) -> BaseSignal:
        """Get signal by type using dictionary-style access."""
        if signal_type not in self.sensors:
            raise KeyError(f"Signal type '{signal_type}' not found")
        return self.sensors[signal_type]
