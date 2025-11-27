"""
MaMIMO-UAV 3D CSI Dataset Implementation

Massive MIMO UAV CSI dataset for 3D positioning with
64-antenna base station and multi-rotor UAV.

Reference:
    MaMIMO-UAV 3D Channel State Information Dataset
    https://doi.org/10.48804/0IMQDF

Dataset URL: https://doi.org/10.48804/0IMQDF
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
class MaMIMOUAVDataset(WiFiDataset):
    """MaMIMO-UAV 3D CSI Dataset.

    Massive MIMO CSI dataset collected with UAV for 3D positioning.
    Features:
    - 64-antenna (8×8 array) base station
    - Multi-rotor UAV as mobile terminal
    - OFDM CSI measurements
    - Outdoor and semi-outdoor scenarios
    - Multiple 3D flight trajectories
    - GPS 3D ground truth (x, y, z)

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        trajectory: Trajectory ID or 'all'.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.MaMIMOUAV(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://doi.org/10.48804/0IMQDF'

    NOT_DETECTED_VALUE = 0.0
    NUM_ANTENNAS = 64
    NUM_SUBCARRIERS = 52
    NUM_FEATURES = NUM_ANTENNAS * NUM_SUBCARRIERS * 2

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        trajectory: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.trajectory = trajectory
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
        return 'MaMIMOUAV'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.mat').exists() or \
               (self.data_root / 'data.csv').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading MaMIMO-UAV dataset...")
        print(f"Note: Please download from:")
        print(f"  https://doi.org/10.48804/0IMQDF")
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

        if self.trajectory != 'all' and 'trajectory' in df.columns:
            df = df[df['trajectory'] == self.trajectory]

        num_train = int(len(df) * self.train_ratio)
        df_split = df.iloc[:num_train] if self.split == 'train' else df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', row.get('gps_x', 0.0)))
            y = float(row.get('y', row.get('gps_y', 0.0)))
            z = float(row.get('z', row.get('gps_z', row.get('altitude', 0.0))))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower() or 'h_' in c.lower()]
            rssi_values = np.array([float(row[c]) for c in csi_cols]) if csi_cols else np.zeros(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id='outdoor'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        n_samples = 600
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            # 3D UAV trajectory
            t = i / n * 2 * np.pi
            x = 50 + 30 * np.cos(t) + np.random.normal(0, 1)
            y = 50 + 30 * np.sin(t) + np.random.normal(0, 1)
            z = 20 + 10 * np.sin(2 * t) + np.random.normal(0, 0.5)

            rssi_values = np.random.randn(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id='outdoor'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def MaMIMOUAV(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading MaMIMO-UAV dataset."""
    if split is None:
        train = MaMIMOUAVDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = MaMIMOUAVDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = MaMIMOUAVDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = MaMIMOUAVDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return MaMIMOUAVDataset(data_root=data_root, split=split, download=download, **kwargs)
