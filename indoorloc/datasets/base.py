"""
Base Dataset Classes for IndoorLoc

Provides abstract base classes for all dataset implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Iterator
from pathlib import Path

from ..signals.base import BaseSignal
from ..locations.location import Location
from ..registry import DATASETS
from ..utils.download import get_data_home


class BaseDataset(ABC):
    """Abstract base class for all indoor localization datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/{dataset_name}).
        split: Dataset split ('train', 'val', 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        # Determine data root
        if data_root is None:
            data_root = get_data_home() / self.dataset_name.lower()
        self.data_root = Path(data_root)

        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.normalize_method = normalize_method

        # Handle download
        if download:
            self._download()

        # Check if data exists
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.data_root}.\n"
                f"Use download=True to download it automatically."
            )

        # Data storage
        self._signals: List[BaseSignal] = []
        self._locations: List[Location] = []
        self._metadata: List[Dict[str, Any]] = []

        # Load data
        self._load_data()

        if self.normalize:
            self._normalize_signals()

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        pass

    @property
    @abstractmethod
    def signal_type(self) -> str:
        """Return the primary signal type of this dataset."""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Load data from files. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _check_exists(self) -> bool:
        """Check if the dataset files exist.

        Returns:
            True if all required files exist, False otherwise.
        """
        pass

    def _download(self) -> None:
        """Download the dataset.

        Subclasses should override this method to implement download logic.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support automatic download.\n"
            f"Please download the dataset manually."
        )

    def _normalize_signals(self) -> None:
        """Normalize all signals using the specified method."""
        self._signals = [
            signal.normalize(method=self.normalize_method)
            for signal in self._signals
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._signals)

    def __getitem__(self, idx: int) -> Tuple[BaseSignal, Location]:
        """Get a single sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (signal, location).
        """
        signal = self._signals[idx]
        location = self._locations[idx]

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, location

    def __iter__(self) -> Iterator[Tuple[BaseSignal, Location]]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self[i]

    @property
    def signals(self) -> List[BaseSignal]:
        """Return all signals."""
        return self._signals

    @property
    def locations(self) -> List[Location]:
        """Return all locations."""
        return self._locations

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Return metadata for all samples."""
        return self._metadata

    def get_statistics(self) -> Dict[str, Any]:
        """Compute and return dataset statistics.

        Returns:
            Dictionary containing dataset statistics.
        """
        import numpy as np

        # Extract coordinates
        xs = [loc.coordinate.x for loc in self._locations]
        ys = [loc.coordinate.y for loc in self._locations]
        floors = [loc.floor for loc in self._locations]
        buildings = [loc.building_id for loc in self._locations]

        return {
            'num_samples': len(self),
            'x_range': (min(xs), max(xs)),
            'y_range': (min(ys), max(ys)),
            'num_floors': len(set(floors)),
            'floors': sorted(set(floors)),
            'num_buildings': len(set(buildings)),
            'buildings': sorted(set(buildings)),
            'signal_type': self.signal_type,
        }

    def split_by_building(self) -> Dict[str, 'BaseDataset']:
        """Split dataset by building.

        Returns:
            Dictionary mapping building_id to subset of data.
        """
        raise NotImplementedError("split_by_building not implemented for this dataset")

    def split_by_floor(self) -> Dict[int, 'BaseDataset']:
        """Split dataset by floor.

        Returns:
            Dictionary mapping floor number to subset of data.
        """
        raise NotImplementedError("split_by_floor not implemented for this dataset")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_root='{self.data_root}', "
            f"split='{self.split}', "
            f"num_samples={len(self)})"
        )


class WiFiDataset(BaseDataset):
    """Base class for WiFi fingerprint datasets.

    Provides common functionality for WiFi RSSI-based datasets.
    """

    @property
    def signal_type(self) -> str:
        return 'wifi'

    @property
    @abstractmethod
    def num_aps(self) -> int:
        """Return the number of access points in this dataset."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Compute WiFi-specific statistics."""
        stats = super().get_statistics()
        stats['num_aps'] = self.num_aps

        # Compute AP detection statistics
        import numpy as np

        detection_counts = []
        for signal in self._signals:
            # Count detected APs (not equal to NOT_DETECTED_VALUE)
            detected = np.sum(signal.rssi_values != 100)
            detection_counts.append(detected)

        stats['avg_detected_aps'] = np.mean(detection_counts)
        stats['min_detected_aps'] = min(detection_counts)
        stats['max_detected_aps'] = max(detection_counts)

        return stats


class BLEDataset(BaseDataset):
    """Base class for BLE (Bluetooth Low Energy) beacon datasets.

    Provides common functionality for BLE RSSI-based indoor localization.
    """

    @property
    def signal_type(self) -> str:
        return 'ble'

    @property
    @abstractmethod
    def num_beacons(self) -> int:
        """Return the number of BLE beacons in this dataset."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Compute BLE-specific statistics."""
        stats = super().get_statistics()
        stats['num_beacons'] = self.num_beacons

        # Compute beacon detection statistics
        import numpy as np

        detection_counts = []
        for signal in self._signals:
            # Count detected beacons (not equal to NOT_DETECTED_VALUE)
            detected = np.sum(signal.rssi_values != 100)
            detection_counts.append(detected)

        stats['avg_detected_beacons'] = np.mean(detection_counts)
        stats['min_detected_beacons'] = min(detection_counts)
        stats['max_detected_beacons'] = max(detection_counts)

        return stats


class UWBDataset(BaseDataset):
    """Base class for UWB (Ultra-Wideband) ranging datasets.

    Provides common functionality for UWB TOF/TDOA-based indoor localization.
    """

    @property
    def signal_type(self) -> str:
        return 'uwb'

    @property
    @abstractmethod
    def num_anchors(self) -> int:
        """Return the number of UWB anchors in this dataset."""
        pass

    @abstractmethod
    def get_anchor_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get 3D positions of all UWB anchors.

        Returns:
            Dictionary mapping anchor IDs to (x, y, z) positions in meters.
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Compute UWB-specific statistics."""
        stats = super().get_statistics()
        stats['num_anchors'] = self.num_anchors

        # Compute distance statistics
        import numpy as np

        all_distances = []
        for signal in self._signals:
            distances = signal.to_numpy()
            all_distances.extend(distances)

        if all_distances:
            stats['avg_distance'] = float(np.mean(all_distances))
            stats['min_distance'] = float(np.min(all_distances))
            stats['max_distance'] = float(np.max(all_distances))
            stats['std_distance'] = float(np.std(all_distances))

        return stats


class HybridDataset(BaseDataset):
    """Base class for hybrid multi-modal sensor datasets.

    Combines multiple signal types (WiFi, BLE, IMU, Magnetometer, etc.)
    for sensor fusion-based indoor localization.
    """

    @property
    def signal_type(self) -> str:
        return 'hybrid'

    @abstractmethod
    def get_signal_modalities(self) -> List[str]:
        """Get list of signal modalities present in this dataset.

        Returns:
            List of signal type strings (e.g., ['wifi', 'magnetometer', 'imu']).
        """
        pass

    def __getitem__(self, idx: int) -> Tuple[BaseSignal, Location]:
        """Get a single sample by index.

        For hybrid datasets, the signal is a HybridSignal containing
        multiple modalities.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (hybrid_signal, location).
        """
        signal = self._signals[idx]
        location = self._locations[idx]

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, location

    def get_statistics(self) -> Dict[str, Any]:
        """Compute hybrid-specific statistics."""
        stats = super().get_statistics()
        stats['modalities'] = self.get_signal_modalities()
        stats['num_modalities'] = len(self.get_signal_modalities())

        # Compute per-modality statistics
        from ..signals.hybrid import HybridSignal

        modality_dims = {}
        for signal in self._signals:
            if isinstance(signal, HybridSignal):
                dims = signal.get_feature_dims_by_modality()
                for modality, dim in dims.items():
                    if modality not in modality_dims:
                        modality_dims[modality] = []
                    modality_dims[modality].append(dim)

        # Average feature dimension per modality
        stats['modality_feature_dims'] = {
            modality: int(sum(dims) / len(dims))
            for modality, dims in modality_dims.items()
        }

        return stats


class MagneticDataset(BaseDataset):
    """Base class for magnetic field (geomagnetic) datasets.

    Provides common functionality for magnetometer-based indoor localization
    using magnetic field fingerprinting.
    """

    @property
    def signal_type(self) -> str:
        return 'magnetometer'

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """Return the sampling rate in Hz.

        Returns:
            Sampling frequency in Hertz.
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Compute magnetic field-specific statistics."""
        stats = super().get_statistics()
        stats['sampling_rate'] = self.sampling_rate

        # Compute magnetic field statistics
        import numpy as np

        all_magnitudes = []
        all_headings = []

        for signal in self._signals:
            magnitudes = signal.compute_magnitude()
            headings = signal.compute_heading()
            all_magnitudes.extend(magnitudes)
            all_headings.extend(headings)

        if all_magnitudes:
            stats['avg_magnitude'] = float(np.mean(all_magnitudes))
            stats['min_magnitude'] = float(np.min(all_magnitudes))
            stats['max_magnitude'] = float(np.max(all_magnitudes))
            stats['std_magnitude'] = float(np.std(all_magnitudes))

        if all_headings:
            stats['avg_heading'] = float(np.mean(all_headings))
            stats['std_heading'] = float(np.std(all_headings))

        return stats
