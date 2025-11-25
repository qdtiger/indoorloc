"""
Base Signal Classes for Indoor Localization

Provides abstract base classes for handling various sensor signals
(WiFi, BLE, IMU, etc.) in a unified manner.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
import numpy as np
import torch


@dataclass
class SignalMetadata:
    """
    Metadata associated with a signal sample.

    Attributes:
        timestamp: Unix timestamp of the signal capture
        device_id: Identifier of the capturing device
        device_type: Type of device (e.g., 'smartphone', 'tablet')
        sampling_rate: Sampling rate in Hz (for time series)
        venue_id: Identifier of the venue/building
        floor_id: Floor number where signal was captured
        extra: Additional metadata as key-value pairs
    """
    timestamp: float = 0.0
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    sampling_rate: Optional[float] = None
    venue_id: Optional[str] = None
    floor_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseSignal(ABC):
    """
    Abstract base class for all signal types.

    All signal implementations (WiFi, BLE, IMU, etc.) should inherit
    from this class and implement the abstract methods.

    Attributes:
        _data: Raw signal data
        _metadata: Signal metadata
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor, Dict],
        metadata: Optional[SignalMetadata] = None
    ):
        """
        Initialize a signal.

        Args:
            data: Signal data (format depends on signal type)
            metadata: Optional metadata for the signal
        """
        self._data = data
        self._metadata = metadata or SignalMetadata()

    @property
    def data(self) -> Union[np.ndarray, torch.Tensor, Dict]:
        """Get raw signal data."""
        return self._data

    @property
    def metadata(self) -> SignalMetadata:
        """Get signal metadata."""
        return self._metadata

    @property
    @abstractmethod
    def signal_type(self) -> str:
        """
        Get the signal type identifier.

        Returns:
            String identifier (e.g., 'wifi', 'ble', 'imu')
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """
        Get the feature dimension of the signal.

        Returns:
            Integer dimension
        """
        pass

    @abstractmethod
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """
        Convert signal to PyTorch tensor.

        Args:
            device: Target device ('cpu' or 'cuda')

        Returns:
            PyTorch tensor representation
        """
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """
        Convert signal to NumPy array.

        Returns:
            NumPy array representation
        """
        pass

    @abstractmethod
    def normalize(self, method: str = 'standard') -> 'BaseSignal':
        """
        Normalize the signal.

        Args:
            method: Normalization method ('standard', 'minmax', etc.)

        Returns:
            New normalized signal instance
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize signal to dictionary.

        Returns:
            Dictionary representation
        """
        data = self._data
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(data, np.ndarray):
            data = data.tolist()

        return {
            'signal_type': self.signal_type,
            'data': data,
            'metadata': {
                'timestamp': self._metadata.timestamp,
                'device_id': self._metadata.device_id,
                'device_type': self._metadata.device_type,
                'venue_id': self._metadata.venue_id,
                'floor_id': self._metadata.floor_id,
                'extra': self._metadata.extra,
            }
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BaseSignal':
        """
        Deserialize signal from dictionary.

        Args:
            d: Dictionary representation

        Returns:
            Signal instance
        """
        pass

    @classmethod
    def collate(cls, signals: List['BaseSignal']) -> torch.Tensor:
        """
        Collate multiple signals into a batch tensor.

        Used by DataLoader for batching.

        Args:
            signals: List of signal instances

        Returns:
            Batched tensor of shape (batch_size, feature_dim)
        """
        tensors = [s.to_tensor() for s in signals]
        return torch.stack(tensors, dim=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"signal_type={self.signal_type}, "
            f"feature_dim={self.feature_dim}, "
            f"timestamp={self._metadata.timestamp})"
        )


__all__ = ['BaseSignal', 'SignalMetadata']
