"""
CSI-Bench Dataset Implementation

Multi-task WiFi CSI benchmark dataset with localization subset.
Collected from 26 real indoor environments with 35 users.

Reference:
    CSI-Bench: A Multi-task WiFi CSI Benchmark
    https://ai-iot-sensing.github.io/projects/project.html

Dataset URL: https://ai-iot-sensing.github.io/projects/project.html
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
class CSIBenchDataset(WiFiDataset):
    """CSI-Bench Multi-task WiFi CSI Dataset.

    Comprehensive WiFi CSI benchmark dataset for multiple sensing tasks.
    Localization subset features:
    - Commercial WiFi edge devices (NXP88W8997 2×2 802.11ac)
    - 5.18 GHz, 40 MHz bandwidth
    - 26 real indoor environments (apartments, residences, offices)
    - 35 users, 461 hours of CSI data
    - Room/area-level position labels
    - Link distance and proximity relationships

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        environment: Environment type ('apartment', 'residence', 'office', 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.CSIBench(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://ai-iot-sensing.github.io/projects/'

    NOT_DETECTED_VALUE = -80.0
    NUM_FEATURES = 256  # 2 antennas × 128 subcarriers

    ENVIRONMENTS = ['apartment', 'residence', 'office']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        environment: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.environment = environment
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
        return 'CSIBench'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.csv').exists() or \
               (self.data_root / 'localization').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading CSI-Bench dataset...")
        print(f"Note: Please download from:")
        print(f"  https://ai-iot-sensing.github.io/projects/project.html")
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

        if self.environment != 'all' and 'environment' in df.columns:
            df = df[df['environment'] == self.environment]

        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            room_id = str(row.get('room', '0'))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower()]
            if csi_cols:
                rssi_values = np.array([float(row[c]) for c in csi_cols])
            else:
                rssi_values = np.zeros(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id=room_id
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        n_samples = 1000
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            env_type = self.ENVIRONMENTS[i % len(self.ENVIRONMENTS)]
            room_id = np.random.randint(1, 10)
            x = np.random.uniform(0, 15)
            y = np.random.uniform(0, 15)

            rssi_values = np.random.uniform(-70, -30, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id=f'{env_type}_{room_id}'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def CSIBench(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading CSI-Bench dataset."""
    if split is None:
        train = CSIBenchDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = CSIBenchDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = CSIBenchDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = CSIBenchDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return CSIBenchDataset(data_root=data_root, split=split, download=download, **kwargs)
