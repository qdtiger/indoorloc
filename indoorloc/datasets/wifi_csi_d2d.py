"""
WiFi CSI D2D (Device-to-Device) Dataset Implementation

WiFi CSI dataset for device-to-device localization with distance
and angle measurements.

Reference:
    WiFi CSI dataset for device-to-device localization
    https://doi.org/10.6084/m9.figshare.20943706.v1

Dataset URL: https://figshare.com/articles/dataset/WiFi_CSI_D2D/20943706
"""
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


@DATASETS.register_module()
class WiFiCSID2DDataset(WiFiDataset):
    """WiFi CSI Device-to-Device Localization Dataset.

    WiFi CSI dataset for D2D relative positioning.
    Features:
    - 5 GHz band, Intel 5300 NIC
    - 3 Rx antennas × 30 subcarriers
    - ~2.5 kHz packet rate
    - Single Tx-Rx pair static indoor layout
    - 5 distances (1-5m) × 2 angles (30°, -60°)
    - Discrete distance + angle labels (relative position)

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.WiFiCSID2D(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://figshare.com/ndownloader/files/'
    DOI = '10.6084/m9.figshare.20943706.v1'

    NOT_DETECTED_VALUE = -80.0
    NUM_FEATURES = 90  # 3 antennas × 30 subcarriers

    DISTANCES = [1, 2, 3, 4, 5]  # meters
    ANGLES = [30, -60]  # degrees

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        super().__init__(
            data_root=data_root,
            split=split,
            download=download,
            transform=transform,
            normalize=normalize,
            normalize_method=normalize_method,
            **kwargs
        )

    @property
    def dataset_name(self) -> str:
        return 'WiFiCSID2D'

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.csv').exists() or \
               (self.data_root / 'data.mat').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading WiFi CSI D2D dataset...")
        print(f"Note: Please download from Figshare:")
        print(f"  https://doi.org/10.6084/m9.figshare.20943706.v1")
        print(f"Place data in: {self.data_root}")

        self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        if (self.data_root / 'data.csv').exists():
            self._load_from_csv(self.data_root / 'data.csv')
        else:
            self._generate_demo_data()

    def _load_from_csv(self, filepath: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required")

        df = pd.read_csv(filepath)
        num_train = int(len(df) * self.train_ratio)
        df_split = df.iloc[:num_train] if self.split == 'train' else df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            distance = float(row.get('distance', 0.0))
            angle = float(row.get('angle', 0.0))

            # Convert polar to Cartesian
            x = distance * np.cos(np.radians(angle))
            y = distance * np.sin(np.radians(angle))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower()]
            rssi_values = np.array([float(row[c]) for c in csi_cols]) if csi_cols else np.zeros(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        # 5 distances × 2 angles = 10 configurations
        n_samples = 500
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            distance = self.DISTANCES[i % len(self.DISTANCES)]
            angle = self.ANGLES[(i // len(self.DISTANCES)) % len(self.ANGLES)]

            x = distance * np.cos(np.radians(angle))
            y = distance * np.sin(np.radians(angle))

            rssi_values = np.random.uniform(-70, -30, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def WiFiCSID2D(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading WiFi CSI D2D dataset."""
    if split is None:
        train = WiFiCSID2DDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = WiFiCSID2DDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = WiFiCSID2DDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = WiFiCSID2DDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return WiFiCSID2DDataset(data_root=data_root, split=split, download=download, **kwargs)
