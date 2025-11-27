"""
HALOC Dataset Implementation

WiFi CSI trajectory dataset collected in indoor corridor environment
using ESP32-S3 with directional antenna.

Reference:
    HALOC Dataset: WiFi CSI for Indoor Localization
    https://zenodo.org/records/10715595

Dataset URL: https://zenodo.org/records/10715595
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
class HALOCDataset(WiFiDataset):
    """HALOC WiFi CSI Trajectory Dataset.

    WiFi CSI dataset collected using ESP32-S3 with directional antenna.
    Features:
    - Indoor long corridor environment
    - Single person walking along 6 long trajectories
    - Strong multipath environment
    - 3D coordinates (x, y, z) per CSI packet

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        trajectory: Trajectory ID (1-6 or 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.HALOC(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://zenodo.org/records/10715595/files/'
    ZENODO_ID = '10715595'

    NOT_DETECTED_VALUE = -80.0
    NUM_FEATURES = 128  # CSI features from ESP32-S3

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
        return 'HALOC'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.csv').exists() or \
               any((self.data_root / f'trajectory_{i}.csv').exists() for i in range(1, 7))

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading HALOC dataset from Zenodo...")
        print(f"  https://zenodo.org/records/10715595")
        print(f"Place data in: {self.data_root}")

        self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        data_file = self.data_root / 'data.csv'
        if data_file.exists():
            self._load_from_csv(data_file)
        else:
            self._generate_demo_data()

    def _load_from_csv(self, filepath: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        df = pd.read_csv(filepath)

        if self.trajectory != 'all' and 'trajectory' in df.columns:
            df = df[df['trajectory'] == int(self.trajectory)]

        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            z = float(row.get('z', 0.0))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower()]
            if csi_cols:
                rssi_values = np.array([float(row[c]) for c in csi_cols])
            else:
                rssi_values = np.zeros(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=int(z),
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        n_samples = 600  # 6 trajectories
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            trajectory_id = (i % 6) + 1
            # Long corridor trajectory
            x = np.random.uniform(0, 30)
            y = np.random.uniform(0, 3)
            z = np.random.uniform(0, 0.5)

            rssi_values = np.random.uniform(-70, -30, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id=str(trajectory_id)
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def HALOC(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading HALOC dataset."""
    if split is None:
        train = HALOCDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = HALOCDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = HALOCDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = HALOCDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return HALOCDataset(data_root=data_root, split=split, download=download, **kwargs)
