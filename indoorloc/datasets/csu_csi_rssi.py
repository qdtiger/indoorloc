"""
CSUIndoorLoc-CSI-RSSI Dataset Implementation

WiFi CSI + RSSI dataset from CSU campus buildings, supporting indoor navigation
with multi-building, multi-floor reference points and trajectory data.

Reference:
    CSUIndoorLoc-CSI-RSSI Dataset
    https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav

Dataset URL: https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav
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
class CSUIndoorLocDataset(WiFiDataset):
    """CSUIndoorLoc-CSI-RSSI Dataset.

    Combined WiFi CSI and RSSI dataset from Colorado State University campus.
    Features:
    - Multiple campus buildings
    - Multi-floor coverage
    - Room and corridor reference points
    - Navigation trajectory data
    - Collected using Nexmon and Atheros CSI tools

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        building: Building to load ('all' or specific building ID).
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.CSUIndoorLoc(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://raw.githubusercontent.com/EPIC-CSU/csi-rssi-dataset-indoor-nav/main'

    NOT_DETECTED_VALUE = -100.0
    NUM_CSI_FEATURES = 256  # CSI subcarriers
    NUM_RSSI_FEATURES = 50  # Number of APs

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        building: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.building = building
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
        return 'CSUIndoorLoc'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return (self.data_root / 'data.csv').exists() or \
               (self.data_root / 'csi_data').exists()

    def _download(self) -> None:
        """Download CSUIndoorLoc dataset."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading CSUIndoorLoc-CSI-RSSI dataset from GitHub...")

        from ..utils.download import download_from_github

        try:
            # Download data files from GitHub
            download_from_github(
                repo='EPIC-CSU/csi-rssi-dataset-indoor-nav',
                root=self.data_root,
                files=['data/csi_data.csv', 'data/rssi_data.csv'],
                branch='main',
            )
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print(f"Please download manually from:")
            print(f"  https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav")
            print(f"Place data in: {self.data_root}")
            self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        """Load CSUIndoorLoc dataset."""
        data_file = self.data_root / 'data.csv'

        if data_file.exists():
            self._load_from_csv(data_file)
        else:
            self._generate_demo_data()

    def _load_from_csv(self, filepath: Path) -> None:
        """Load data from CSV file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        df = pd.read_csv(filepath)

        if self.building != 'all' and 'building' in df.columns:
            df = df[df['building'] == self.building]

        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            floor = int(row.get('floor', 0))
            building_id = str(row.get('building', '0'))

            # Combine CSI and RSSI features
            csi_cols = [c for c in df_split.columns if 'csi' in c.lower()]
            rssi_cols = [c for c in df_split.columns if 'rssi' in c.lower() or 'wap' in c.lower()]

            features = []
            for col in csi_cols + rssi_cols:
                val = row[col]
                if pd.isna(val):
                    val = self.NOT_DETECTED_VALUE
                features.append(float(val))

            if not features:
                features = np.zeros(self.NUM_CSI_FEATURES + self.NUM_RSSI_FEATURES)

            signal = WiFiSignal(rssi_values=np.array(features))
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=floor,
                building_id=building_id
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        """Generate demonstration data."""
        np.random.seed(42 if self.split == 'train' else 123)

        n_samples = 500
        num_train = int(n_samples * self.train_ratio)
        if self.split == 'train':
            n = num_train
        else:
            n = n_samples - num_train

        buildings = ['B1', 'B2', 'B3']
        floors = [0, 1, 2]

        for _ in range(n):
            x = np.random.uniform(0, 50)
            y = np.random.uniform(0, 50)
            floor = np.random.choice(floors)
            building_id = np.random.choice(buildings)

            # Combined CSI + RSSI features
            csi_features = np.random.uniform(-70, -30, self.NUM_CSI_FEATURES)
            rssi_features = np.random.uniform(-90, -30, self.NUM_RSSI_FEATURES)
            features = np.concatenate([csi_features, rssi_features])

            signal = WiFiSignal(rssi_values=features)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=floor,
                building_id=building_id
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def CSUIndoorLoc(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading CSUIndoorLoc-CSI-RSSI dataset."""
    if split is None:
        train_dataset = CSUIndoorLocDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = CSUIndoorLocDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = CSUIndoorLocDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = CSUIndoorLocDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return CSUIndoorLocDataset(
            data_root=data_root, split=split, download=download, **kwargs
        )
