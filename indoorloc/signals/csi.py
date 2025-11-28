"""
CSI (Channel State Information) Signal Implementation

Handles complex-valued CSI data for indoor localization,
commonly used in MIMO wireless systems like DeepMIMO.
"""
from typing import Dict, Optional, Any, Union, Literal
from dataclasses import dataclass
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@dataclass
class CSIMetadata(SignalMetadata):
    """
    Extended metadata for CSI signals.

    Attributes:
        num_antennas: Number of antennas in the MIMO system
        num_subcarriers: Number of OFDM subcarriers
        frequency_ghz: Carrier frequency in GHz
        bandwidth_mhz: Bandwidth in MHz
        scenario: Scenario name (e.g., 'O1_60', 'I3_60')
    """
    num_antennas: Optional[int] = None
    num_subcarriers: Optional[int] = None
    frequency_ghz: Optional[float] = None
    bandwidth_mhz: Optional[float] = None
    scenario: Optional[str] = None


@SIGNALS.register_module()
class CSISignal(BaseSignal):
    """
    Complex-valued CSI Signal for MIMO-based localization.

    Stores CSI data as complex numbers and provides multiple
    representations for neural network input.

    The CSI matrix can have various shapes depending on the system:
    - (num_subcarriers,): Single antenna
    - (num_antennas, num_subcarriers): MIMO system
    - (num_rx, num_tx, num_subcarriers): Full MIMO channel matrix

    Attributes:
        _csi: Complex-valued CSI data (complex64 or complex128)
    """

    def __init__(
        self,
        csi_values: Union[np.ndarray, torch.Tensor],
        metadata: Optional[CSIMetadata] = None
    ):
        """
        Initialize a CSI signal.

        Args:
            csi_values: Complex CSI values as numpy array or torch tensor.
                        Will be converted to complex64 if not already complex.
            metadata: Optional CSI metadata
        """
        # Convert to numpy if torch tensor
        if isinstance(csi_values, torch.Tensor):
            csi_values = csi_values.numpy()

        # Ensure complex type
        if not np.iscomplexobj(csi_values):
            # If real, assume it's magnitude and create complex with zero phase
            csi_values = csi_values.astype(np.complex64)
        else:
            csi_values = csi_values.astype(np.complex64)

        self._csi = csi_values
        super().__init__(csi_values, metadata or CSIMetadata())

    @property
    def signal_type(self) -> str:
        return 'csi'

    @property
    def feature_dim(self) -> int:
        """
        Get feature dimension (total number of complex values).
        """
        return self._csi.size

    @property
    def shape(self) -> tuple:
        """Get the shape of the CSI matrix."""
        return self._csi.shape

    @property
    def num_antennas(self) -> int:
        """
        Get number of antennas (inferred from shape).

        For 1D: returns 1
        For 2D: returns shape[0]
        For 3D: returns shape[0] * shape[1] (rx * tx)
        """
        if self._csi.ndim == 1:
            return 1
        elif self._csi.ndim == 2:
            return self._csi.shape[0]
        else:
            return self._csi.shape[0] * self._csi.shape[1]

    @property
    def num_subcarriers(self) -> int:
        """Get number of subcarriers."""
        return self._csi.shape[-1]

    @property
    def csi_complex(self) -> np.ndarray:
        """Get raw complex CSI data."""
        return self._csi

    @property
    def as_real_imag(self) -> np.ndarray:
        """
        Convert to real representation by stacking real and imaginary parts.

        Returns:
            Array with real and imag concatenated along last axis.
            Shape: (..., 2*num_subcarriers) or (..., num_subcarriers, 2)
        """
        return np.stack([self._csi.real, self._csi.imag], axis=-1)

    @property
    def as_magnitude_phase(self) -> np.ndarray:
        """
        Convert to magnitude/phase representation.

        Returns:
            Array with magnitude and phase stacked along last axis.
            Shape: (..., num_subcarriers, 2)
        """
        return np.stack([np.abs(self._csi), np.angle(self._csi)], axis=-1)

    @property
    def magnitude(self) -> np.ndarray:
        """Get magnitude (absolute value) of CSI."""
        return np.abs(self._csi)

    @property
    def phase(self) -> np.ndarray:
        """Get phase (angle) of CSI in radians."""
        return np.angle(self._csi)

    def flatten(self) -> 'CSISignal':
        """
        Flatten CSI to 1D array.

        Returns:
            New CSISignal with flattened data
        """
        return CSISignal(
            csi_values=self._csi.flatten(),
            metadata=self._metadata
        )

    def to_tensor(
        self,
        device: str = 'cpu',
        representation: Literal['real_imag', 'magnitude_phase', 'complex'] = 'real_imag'
    ) -> torch.Tensor:
        """
        Convert to PyTorch tensor.

        Args:
            device: Target device ('cpu' or 'cuda')
            representation: How to represent complex values
                - 'real_imag': Stack real and imag parts (default)
                - 'magnitude_phase': Stack magnitude and phase
                - 'complex': Return as torch.complex64 (requires PyTorch >= 1.6)

        Returns:
            PyTorch tensor
        """
        if representation == 'complex':
            return torch.tensor(self._csi, dtype=torch.complex64, device=device)
        elif representation == 'real_imag':
            data = self.as_real_imag
        elif representation == 'magnitude_phase':
            data = self.as_magnitude_phase
        else:
            raise ValueError(f"Unknown representation: {representation}")

        return torch.tensor(data, dtype=torch.float32, device=device)

    def to_numpy(
        self,
        representation: Literal['real_imag', 'magnitude_phase', 'complex'] = 'complex'
    ) -> np.ndarray:
        """
        Convert to NumPy array.

        Args:
            representation: How to represent complex values
                - 'complex': Return as complex64 (default)
                - 'real_imag': Stack real and imag parts
                - 'magnitude_phase': Stack magnitude and phase

        Returns:
            NumPy array
        """
        if representation == 'complex':
            return self._csi.copy()
        elif representation == 'real_imag':
            return self.as_real_imag
        elif representation == 'magnitude_phase':
            return self.as_magnitude_phase
        else:
            raise ValueError(f"Unknown representation: {representation}")

    def normalize(
        self,
        method: Literal['standard', 'minmax', 'unit'] = 'standard'
    ) -> 'CSISignal':
        """
        Normalize the CSI signal.

        Args:
            method: Normalization method
                - 'standard': Z-score normalization (per real/imag)
                - 'minmax': Scale magnitude to [0, 1]
                - 'unit': Normalize each subcarrier to unit magnitude

        Returns:
            New normalized CSISignal
        """
        csi = self._csi.copy()

        if method == 'standard':
            # Normalize real and imaginary parts separately
            real = csi.real
            imag = csi.imag
            real = (real - real.mean()) / (real.std() + 1e-8)
            imag = (imag - imag.mean()) / (imag.std() + 1e-8)
            csi = real + 1j * imag

        elif method == 'minmax':
            # Scale magnitude to [0, 1], preserve phase
            mag = np.abs(csi)
            phase = np.angle(csi)
            mag_min, mag_max = mag.min(), mag.max()
            mag_norm = (mag - mag_min) / (mag_max - mag_min + 1e-8)
            csi = mag_norm * np.exp(1j * phase)

        elif method == 'unit':
            # Normalize each element to unit magnitude
            mag = np.abs(csi)
            csi = csi / (mag + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return CSISignal(csi_values=csi, metadata=self._metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize signal to dictionary."""
        return {
            'signal_type': self.signal_type,
            'data': {
                'real': self._csi.real.tolist(),
                'imag': self._csi.imag.tolist(),
            },
            'shape': self._csi.shape,
            'metadata': {
                'timestamp': self._metadata.timestamp,
                'device_id': self._metadata.device_id,
                'venue_id': self._metadata.venue_id,
                'floor_id': self._metadata.floor_id,
                'extra': self._metadata.extra,
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CSISignal':
        """Create CSISignal from dictionary."""
        real = np.array(d['data']['real'])
        imag = np.array(d['data']['imag'])
        csi = real + 1j * imag

        metadata = CSIMetadata(
            timestamp=d.get('metadata', {}).get('timestamp', 0.0),
            device_id=d.get('metadata', {}).get('device_id'),
            venue_id=d.get('metadata', {}).get('venue_id'),
            floor_id=d.get('metadata', {}).get('floor_id'),
            extra=d.get('metadata', {}).get('extra', {}),
        )

        return cls(csi_values=csi, metadata=metadata)

    @classmethod
    def from_real_imag(
        cls,
        real: np.ndarray,
        imag: np.ndarray,
        metadata: Optional[CSIMetadata] = None
    ) -> 'CSISignal':
        """
        Create CSISignal from separate real and imaginary arrays.

        Args:
            real: Real part of CSI
            imag: Imaginary part of CSI
            metadata: Optional metadata

        Returns:
            CSISignal instance
        """
        csi = real.astype(np.float32) + 1j * imag.astype(np.float32)
        return cls(csi_values=csi, metadata=metadata)

    @classmethod
    def from_magnitude_phase(
        cls,
        magnitude: np.ndarray,
        phase: np.ndarray,
        metadata: Optional[CSIMetadata] = None
    ) -> 'CSISignal':
        """
        Create CSISignal from magnitude and phase arrays.

        Args:
            magnitude: Magnitude of CSI
            phase: Phase of CSI in radians
            metadata: Optional metadata

        Returns:
            CSISignal instance
        """
        csi = magnitude * np.exp(1j * phase)
        return cls(csi_values=csi.astype(np.complex64), metadata=metadata)

    def __len__(self) -> int:
        return self.feature_dim

    def __repr__(self) -> str:
        return (
            f"CSISignal(shape={self.shape}, "
            f"num_antennas={self.num_antennas}, "
            f"num_subcarriers={self.num_subcarriers})"
        )


__all__ = ['CSISignal', 'CSIMetadata']
