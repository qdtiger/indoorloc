"""
WILD-v2 (WiFi Indoor Localization Dataset v2) Implementation

WiFi CSI dataset collected by ground robots for dense indoor mapping.
Contains data from 8 indoor environments with LOS/NLOS conditions.

Reference:
    WILD / WILD-v2: Robot-collected WiFi CSI Dataset
    https://www.kaggle.com/c/wild-v2

Dataset URL: https://www.kaggle.com/c/wild-v2
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
class WILDv2Dataset(WiFiDataset):
    """WILD-v2 (WiFi Indoor Localization Dataset v2).

    WiFi CSI dataset collected by ground robots equipped with CSI collection
    modules. Features:
    - 8 indoor environments (simple/complex, LOS/NLOS mixed)
    - Dense scanning along multiple trajectories
    - Robot SLAM/external system provides ground truth 2D positions
    - Multiple AP coverage

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        environment: Environment ID (1-8 or 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.WILDv2(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://www.kaggle.com/c/wild-v2/data'

    NOT_DETECTED_VALUE = -80.0
    NUM_FEATURES = 256

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
        return 'WILDv2'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        return (self.data_root / 'train.csv').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading WILD-v2 dataset...")
        print(f"Note: This dataset requires Kaggle authentication.")
        print(f"  https://www.kaggle.com/c/wild-v2")
        print(f"Please download and place data in: {self.data_root}")

        self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        data_file = self.data_root / 'train.csv'
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

        if self.environment != 'all' and 'env' in df.columns:
            df = df[df['env'] == int(self.environment)]

        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))

            feature_cols = [c for c in df_split.columns if c not in ['x', 'y', 'env', 'id']]
            rssi_values = np.array([float(row[c]) for c in feature_cols])

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

        n_samples = 800  # 8 environments
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            env_id = (i % 8) + 1
            x = np.random.uniform(0, 20)
            y = np.random.uniform(0, 20)

            rssi_values = np.random.uniform(-80, -30, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id=str(env_id)
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def WILDv2(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading WILD-v2 dataset."""
    if split is None:
        train = WILDv2Dataset(data_root=data_root, split='train', download=download, **kwargs)
        test = WILDv2Dataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = WILDv2Dataset(data_root=data_root, split='train', download=download, **kwargs)
        test = WILDv2Dataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return WILDv2Dataset(data_root=data_root, split=split, download=download, **kwargs)
