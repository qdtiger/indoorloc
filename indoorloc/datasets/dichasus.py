"""
DICHASUS Massive MIMO CSI Collection Implementation

Distributed Massive MIMO channel measurement collection with
multiple scenarios including indoor, outdoor, and cell edge.

Reference:
    DICHASUS Massive MIMO CSI Collection
    https://darus.uni-stuttgart.de/dataverse/dichasus

Dataset URL: https://darus.uni-stuttgart.de/dataverse/dichasus
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
class DICHASUSDataset(WiFiDataset):
    """DICHASUS Massive MIMO CSI Collection.

    OFDM-based Massive MIMO measurement platform collection.
    Features:
    - Multi-frequency band support
    - Distributed and co-located array configurations
    - Multiple scenarios (indoor LoS/NLoS, outdoor, cell edge)
    - Various UE motion trajectories
    - 2D/3D ground truth positions per sub-dataset

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        scenario: Scenario type ('indoor_los', 'indoor_nlos', 'outdoor', 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.DICHASUS(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://darus.uni-stuttgart.de/dataverse/dichasus'

    NOT_DETECTED_VALUE = 0.0
    NUM_FEATURES = 2048

    SCENARIOS = ['indoor_los', 'indoor_nlos', 'outdoor', 'cell_edge']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        scenario: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.scenario = scenario
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
        return 'DICHASUS'

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

        print(f"Downloading DICHASUS dataset...")
        print(f"Note: Please download from DaRUS:")
        print(f"  https://darus.uni-stuttgart.de/dataverse/dichasus")
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

        if self.scenario != 'all' and 'scenario' in df.columns:
            df = df[df['scenario'] == self.scenario]

        num_train = int(len(df) * self.train_ratio)
        df_split = df.iloc[:num_train] if self.split == 'train' else df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            z = float(row.get('z', 0.0))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower() or 'h_' in c.lower()]
            rssi_values = np.array([float(row[c]) for c in csi_cols]) if csi_cols else np.zeros(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id='0'
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
            scenario = self.SCENARIOS[i % len(self.SCENARIOS)]
            x = np.random.uniform(0, 20)
            y = np.random.uniform(0, 20)
            z = np.random.uniform(0, 3)

            rssi_values = np.random.randn(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id=scenario
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def DICHASUS(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading DICHASUS dataset."""
    if split is None:
        train = DICHASUSDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = DICHASUSDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = DICHASUSDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = DICHASUSDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return DICHASUSDataset(data_root=data_root, split=split, download=download, **kwargs)
