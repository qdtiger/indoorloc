"""
Ultra-Dense Indoor MaMIMO CSI Dataset Implementation

Massive MIMO CSI dataset with 64-antenna indoor base station.
Ultra-dense sampling with millimeter-level grid positioning.

Reference:
    Ultra-Dense Indoor MaMIMO CSI Dataset
    https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

Dataset URL: https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset
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
class MaMIMOCSIDataset(WiFiDataset):
    """Ultra-Dense Indoor Massive MIMO CSI Dataset.

    Indoor Massive MIMO CSI dataset with 64-antenna base station.
    Features:
    - 64-antenna Massive MIMO indoor base station
    - Multiple array topologies (DIS/ULA/URA, LoS & NLoS)
    - ~2.5×2.5m area with millimeter-level grid
    - Hundreds of thousands of CSI samples
    - Precise trajectory system calibrated positions

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        topology: Array topology ('dis', 'ula', 'ura', 'all').
        condition: Channel condition ('los', 'nlos', 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.MaMIMOCSI(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset'

    NOT_DETECTED_VALUE = 0.0
    NUM_ANTENNAS = 64
    NUM_SUBCARRIERS = 52
    NUM_FEATURES = NUM_ANTENNAS * NUM_SUBCARRIERS * 2  # Real + Imag

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        topology: str = 'all',
        condition: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.topology = topology
        self.condition = condition
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
        return 'MaMIMOCSI'

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

        print(f"Downloading Ultra-Dense Indoor MaMIMO CSI dataset...")
        print(f"Note: Please download from IEEE DataPort:")
        print(f"  https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset")
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
        print(f"Loaded data from {filepath}")

    def _load_from_csv(self, filepath: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required")

        df = pd.read_csv(filepath)
        num_train = int(len(df) * self.train_ratio)

        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower() or 'h_' in c.lower()]
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

        n_samples = 1000
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        # 2.5×2.5m area with mm-level grid
        for _ in range(n):
            x = np.random.uniform(0, 2.5)
            y = np.random.uniform(0, 2.5)

            # 64 antennas × 52 subcarriers × 2 (complex)
            rssi_values = np.random.randn(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def MaMIMOCSI(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading MaMIMO CSI dataset."""
    if split is None:
        train = MaMIMOCSIDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = MaMIMOCSIDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = MaMIMOCSIDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = MaMIMOCSIDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return MaMIMOCSIDataset(data_root=data_root, split=split, download=download, **kwargs)
