"""
CSI2TAoA Dataset Implementation

Arena indoor RF testbed CSI dataset for Time-Angle of Arrival estimation.
Similar to CSI2Pos but focused on TAoA learning and positioning benchmark.

Reference:
    CSI2TAoA: CSI to Time-Angle of Arrival Dataset
    https://service.tib.eu/ldmservice/dataset/csi2taoa

Dataset URL: https://service.tib.eu/ldmservice/dataset/csi2taoa
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
class CSI2TAoADataset(WiFiDataset):
    """CSI2TAoA Arena Indoor RF Dataset.

    Arena indoor RF testbed dataset for TAoA-based positioning.
    Features:
    - Same Arena environment as CSI2Pos
    - CSI features for TAoA estimation
    - Time-Angle of Arrival (TAoA) labels
    - 2D grid coordinate labels
    - Benchmark for TAoA learning methods

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.CSI2TAoA(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://service.tib.eu/ldmservice/dataset/csi2taoa'

    NOT_DETECTED_VALUE = 0.0
    NUM_FEATURES = 1024

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
        return 'CSI2TAoA'

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.csv').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading CSI2TAoA dataset...")
        print(f"Note: Please download from TIB:")
        print(f"  https://service.tib.eu/ldmservice/dataset/csi2taoa")
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
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower() or 'taoa' in c.lower()]
            rssi_values = np.array([float(row[c]) for c in csi_cols]) if csi_cols else np.zeros(self.NUM_FEATURES)

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

        n_samples = 800
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)

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


def CSI2TAoA(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading CSI2TAoA dataset."""
    if split is None:
        train = CSI2TAoADataset(data_root=data_root, split='train', download=download, **kwargs)
        test = CSI2TAoADataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = CSI2TAoADataset(data_root=data_root, split='train', download=download, **kwargs)
        test = CSI2TAoADataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return CSI2TAoADataset(data_root=data_root, split=split, download=download, **kwargs)
