"""
H-WILD (Human-held WiFi Indoor Localization Dataset) Implementation

WiFi CSI dataset collected with human-held devices, providing realistic
indoor localization data with UWB ground truth coordinates.

Reference:
    H-WILD: Human-held device WiFi indoor localization dataset
    https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset

Dataset URL: https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset
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
class HWILDDataset(WiFiDataset):
    """H-WILD (Human-held WiFi Indoor Localization Dataset).

    WiFi CSI dataset collected using commercial WiFi APs with human-held
    terminal devices. Features:
    - CSI dimension: ~1×90 (3 antennas × 30 subcarriers)
    - 4 indoor environments: Conference, Laboratory, Office, Lounge
    - 10 volunteers with diverse holding positions
    - ~120,000 samples total
    - UWB-provided ground truth coordinates

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        environment: Indoor environment ('conference', 'laboratory', 'office', 'lounge', 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.HWILD(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://raw.githubusercontent.com/H-WILD/human_held_device_wifi_indoor_localization_dataset/main'

    NOT_DETECTED_VALUE = -80.0
    NUM_FEATURES = 90  # 3 antennas × 30 subcarriers

    ENVIRONMENTS = ['conference', 'laboratory', 'office', 'lounge']

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
        return 'HWILD'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return (self.data_root / 'data.csv').exists() or \
               any((self.data_root / env).exists() for env in self.ENVIRONMENTS)

    def _download(self) -> None:
        """Download H-WILD dataset."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading H-WILD dataset...")
        print(f"Note: This dataset requires manual download from:")
        print(f"  https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset")
        print(f"Please download and place data in: {self.data_root}")

        self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        """Load H-WILD dataset."""
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

        # Filter by environment if specified
        if self.environment != 'all' and 'environment' in df.columns:
            df = df[df['environment'] == self.environment]

        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('uwb_coordinate_x', row.get('x', 0.0)))
            y = float(row.get('uwb_coordinate_y', row.get('y', 0.0)))

            csi_cols = [c for c in df_split.columns if 'csi' in c.lower()]
            if csi_cols:
                rssi_values = np.array([float(row[c]) for c in csi_cols])
            else:
                rssi_values = np.zeros(self.NUM_FEATURES)

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

        # ~120,000 samples total, distribute across environments
        samples_per_env = {
            'conference': 30000,
            'laboratory': 30000,
            'office': 30000,
            'lounge': 30000
        }

        if self.environment == 'all':
            n_samples = sum(samples_per_env.values())
        else:
            n_samples = samples_per_env.get(self.environment, 30000)

        # Scale down for demo
        n_samples = min(n_samples, 1000)

        num_train = int(n_samples * self.train_ratio)
        if self.split == 'train':
            n = num_train
        else:
            n = n_samples - num_train

        for _ in range(n):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)

            # 3 antennas × 30 subcarriers = 90 CSI features
            rssi_values = np.random.uniform(-70, -30, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")
        print(f"Note: Download actual data from GitHub repository")


def HWILD(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading H-WILD dataset.

    Returns:
        - If split is None: Returns tuple (train_dataset, test_dataset)
        - If split is 'all': Returns merged train+test dataset
        - Otherwise: Returns single dataset for specified split
    """
    if split is None:
        train_dataset = HWILDDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = HWILDDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = HWILDDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = HWILDDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return HWILDDataset(
            data_root=data_root, split=split, download=download, **kwargs
        )
