"""
CSI2Pos Dataset Implementation

Arena indoor RF testbed CSI dataset for position estimation.
Contains CSI and periodogram (PER) features mapped to 2D grid coordinates.

Reference:
    CSI2Pos: CSI to Position Dataset
    https://service.tib.eu/ldmservice/dataset/csi2pos

Dataset URL: https://service.tib.eu/ldmservice/dataset/csi2pos
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
class CSI2PosDataset(WiFiDataset):
    """CSI2Pos Arena Indoor RF Dataset.

    Arena indoor RF testbed dataset for CSI-based positioning.
    Features:
    - Arena1 indoor environment
    - ~2,700 spatial grid positions
    - ~53,000 total samples
    - CSI and periodogram (PER) vectors
    - 2D grid coordinate labels

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        feature_type: Feature type ('csi', 'per', 'both').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.CSI2Pos(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://service.tib.eu/ldmservice/dataset/csi2pos'

    NOT_DETECTED_VALUE = 0.0
    NUM_FEATURES = 1024

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        feature_type: str = 'csi',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.feature_type = feature_type
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
        return 'CSI2Pos'

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.csv').exists() or \
               (self.data_root / 'csi2pos.mat').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading CSI2Pos dataset...")
        print(f"Note: Please download from TIB:")
        print(f"  https://service.tib.eu/ldmservice/dataset/csi2pos")
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
            x = float(row.get('x', row.get('grid_x', 0.0)))
            y = float(row.get('y', row.get('grid_y', 0.0)))

            if self.feature_type == 'csi':
                cols = [c for c in df_split.columns if 'csi' in c.lower()]
            elif self.feature_type == 'per':
                cols = [c for c in df_split.columns if 'per' in c.lower()]
            else:
                cols = [c for c in df_split.columns if 'csi' in c.lower() or 'per' in c.lower()]

            rssi_values = np.array([float(row[c]) for c in cols]) if cols else np.zeros(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='arena1'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        # ~2700 grid positions, ~53000 samples (20 per position)
        n_samples = 1000
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        # Grid-based positions
        grid_size = int(np.sqrt(2700))
        for i in range(n):
            grid_x = i % grid_size
            grid_y = i // grid_size
            x = grid_x * 0.1  # 10cm grid
            y = grid_y * 0.1

            rssi_values = np.random.randn(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='arena1'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def CSI2Pos(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading CSI2Pos dataset."""
    if split is None:
        train = CSI2PosDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = CSI2PosDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = CSI2PosDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = CSI2PosDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return CSI2PosDataset(data_root=data_root, split=split, download=download, **kwargs)
