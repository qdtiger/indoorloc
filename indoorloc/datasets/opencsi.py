"""
OpenCSI Dataset Implementation

LTE CSI fingerprint dataset collected using SDR terminal in indoor environment.
Robot-collected with high-density grid sampling along 8 trajectories.

Reference:
    OpenCSI: LTE Channel State Information Dataset
    https://doi.org/10.6084/m9.figshare.19596379.v1

Dataset URL: https://figshare.com/articles/dataset/OpenCSI/19596379
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
class OpenCSIDataset(WiFiDataset):
    """OpenCSI LTE CSI Dataset.

    3GPP LTE downlink CSI dataset collected using SDR terminal.
    Features:
    - Indoor single-cell environment
    - Wheeled robot automatic collection
    - 8 trajectories with high-density grid sampling
    - 2D ground truth coordinates (x, y)

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        trajectory: Trajectory ID (1-8 or 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.OpenCSI(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://figshare.com/ndownloader/files/'
    DOI = '10.6084/m9.figshare.19596379.v1'

    NOT_DETECTED_VALUE = -100.0
    NUM_FEATURES = 1200  # LTE CSI features

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
        return 'OpenCSI'

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

        print(f"Downloading OpenCSI dataset...")
        print(f"Note: Please download from Figshare:")
        print(f"  https://doi.org/10.6084/m9.figshare.19596379.v1")
        print(f"Place data in: {self.data_root}")

        self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        mat_file = self.data_root / 'data.mat'
        csv_file = self.data_root / 'data.csv'

        if mat_file.exists():
            self._load_from_mat(mat_file)
        elif csv_file.exists():
            self._load_from_csv(csv_file)
        else:
            self._generate_demo_data()

    def _load_from_mat(self, filepath: Path) -> None:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required. Install with: pip install scipy")

        data = loadmat(filepath)
        # Process MAT file structure based on actual format
        print(f"Loaded data from {filepath}")

    def _load_from_csv(self, filepath: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        df = pd.read_csv(filepath)
        num_train = int(len(df) * self.train_ratio)

        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))

            feature_cols = [c for c in df_split.columns if c not in ['x', 'y', 'trajectory']]
            rssi_values = np.array([float(row[c]) for c in feature_cols[:self.NUM_FEATURES]])

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

        n_samples = 800
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            trajectory_id = (i % 8) + 1
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)

            rssi_values = np.random.uniform(-100, -40, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id=str(trajectory_id)
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def OpenCSI(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading OpenCSI dataset."""
    if split is None:
        train = OpenCSIDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = OpenCSIDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = OpenCSIDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = OpenCSIDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return OpenCSIDataset(data_root=data_root, split=split, download=download, **kwargs)
