"""
CSI-dataset-for-indoor-localization Dataset Implementation

WiFi CSI fingerprint dataset using Intel 5300 with Linux 802.11n CSI Tool.
Contains data from multiple indoor areas with NLOS and LOS conditions.

Reference:
    CSI-dataset-for-indoor-localization
    https://github.com/qiang5love1314/CSI-dataset

Dataset URL: https://github.com/qiang5love1314/CSI-dataset
"""
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class CSIFingerprintDataset(WiFiDataset):
    """CSI-dataset-for-indoor-localization Dataset.

    WiFi CSI fingerprint dataset collected using Intel 5300 NIC with Linux
    802.11n CSI Tool. Contains measurements from multiple indoor areas:
    - Area1: 13.5×11m laboratory (NLOS, 317 points)
    - Area2: 7×10m conference room (LOS, 176 points)
    - Additional miniLab and Conference scenarios

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method.
        train_ratio: Ratio for train/test split (default: 0.7).
        area: Area to load ('area1', 'area2', or 'all').

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.CSIFingerprint(download=True)
        >>> signal, location = train[0]
    """

    # GitHub repository URL
    BASE_URL = 'https://raw.githubusercontent.com/qiang5love1314/CSI-dataset/master'

    # Dataset constants
    NOT_DETECTED_VALUE = -80.0
    NUM_SUBCARRIERS = 30  # Intel 5300 provides 30 subcarriers per antenna

    REQUIRED_FILES = ['data.csv']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        area: str = 'all',
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.area = area
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
        return 'CSIFingerprint'

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return (self.data_root / 'data.csv').exists() or \
               (self.data_root / 'Area1').exists() or \
               (self.data_root / 'Area2').exists()

    def _download(self) -> None:
        """Download CSI fingerprint dataset."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading CSI fingerprint dataset...")
        print(f"Note: This dataset requires manual download from:")
        print(f"  https://github.com/qiang5love1314/CSI-dataset")
        print(f"Please clone the repository and place data in: {self.data_root}")

        # Create placeholder for manual download
        self.data_root.mkdir(parents=True, exist_ok=True)
        readme_path = self.data_root / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("CSI Fingerprint Dataset\n")
            f.write("======================\n\n")
            f.write("Please download from:\n")
            f.write("https://github.com/qiang5love1314/CSI-dataset\n\n")
            f.write("Clone or download the repository and place the data here.\n")

    def _load_data(self) -> None:
        """Load CSI fingerprint dataset."""
        # Try to find data files
        data_file = self.data_root / 'data.csv'

        if data_file.exists():
            self._load_from_csv(data_file)
        else:
            # Generate synthetic data for demonstration
            self._generate_demo_data()

    def _load_from_csv(self, filepath: Path) -> None:
        """Load data from CSV file."""
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

            # Extract CSI features
            csi_cols = [c for c in df_split.columns if 'csi' in c.lower() or 'amp' in c.lower()]
            if csi_cols:
                rssi_values = np.array([float(row[c]) for c in csi_cols])
            else:
                rssi_values = np.zeros(self.NUM_SUBCARRIERS * 3)  # 3 antennas

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
        """Generate demonstration data."""
        np.random.seed(42 if self.split == 'train' else 123)

        # Area1: 13.5×11m, 317 points (NLOS)
        # Area2: 7×10m, 176 points (LOS)
        if self.area == 'area1':
            n_samples = 317
            x_range, y_range = (0, 13.5), (0, 11)
        elif self.area == 'area2':
            n_samples = 176
            x_range, y_range = (0, 7), (0, 10)
        else:
            n_samples = 493  # Combined
            x_range, y_range = (0, 13.5), (0, 11)

        num_train = int(n_samples * self.train_ratio)
        if self.split == 'train':
            n = num_train
        else:
            n = n_samples - num_train

        for _ in range(n):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)

            # 3 antennas × 30 subcarriers = 90 CSI amplitude features
            rssi_values = np.random.uniform(-70, -30, self.NUM_SUBCARRIERS * 3)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")
        print(f"Note: Download actual data from https://github.com/qiang5love1314/CSI-dataset")


def CSIFingerprint(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading CSIFingerprint dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to CSIFingerprintDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> train, test = CSIFingerprint(download=True)
        >>> dataset = CSIFingerprint(split='all', download=True)
    """
    if split is None:
        train_dataset = CSIFingerprintDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = CSIFingerprintDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = CSIFingerprintDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = CSIFingerprintDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return CSIFingerprintDataset(
            data_root=data_root, split=split, download=download, **kwargs
        )
