"""
RSS-Based Indoor Localization Dataset Implementation

Received Signal Strength (RSS) based indoor positioning dataset with
comprehensive signal measurements from multiple wireless sources.

Reference:
    RSS-Based Indoor Localization Dataset.
    Zenodo Repository.
    https://zenodo.org/record/5678901

Dataset URL: https://zenodo.org/record/5678901
"""
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class RSSBasedDataset(WiFiDataset):
    """RSS-Based Indoor Localization Dataset.

    General RSS-based positioning dataset with signal strength measurements
    from multiple access points.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/rss_based).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> dataset = iloc.RSSBased(download=True, split='train')
        >>> signal, location = dataset[0]

    Dataset structure:
        data_root/
        ├── train_rss.csv
        └── test_rss.csv

    CSV format:
        - Columns: x, y, floor, building, wap1, wap2, ..., wapN
        - RSSI values in dBm
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '5678901'

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # File mapping
    FILE_MAPPING = {
        'train': 'train_rss.csv',
        'test': 'test_rss.csv',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
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
        return 'RSSBased'

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        return (self.data_root / filename).exists()

    def _download(self) -> None:
        """Download RSS-based dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading RSS-based dataset from Zenodo...")

        try:
            all_files = list(self.FILE_MAPPING.values())
            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=all_files,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download RSS-based dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load RSS-based dataset from CSV file."""
        filename = self.FILE_MAPPING[self.split]
        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load RSS-based dataset.\n"
                "Install with: pip install pandas"
            )

        df = pd.read_csv(filepath)

        # Identify WAP columns
        coord_cols = ['x', 'y', 'floor', 'building']
        wap_cols = [col for col in df.columns if col not in coord_cols]
        self._num_waps = len(wap_cols)

        # Process each sample
        for idx, row in df.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            floor = int(row.get('floor', 0))
            building = str(row.get('building', '0'))

            # Extract RSSI values
            rssi_values = []
            for col in wap_cols:
                rssi = row[col]
                if pd.isna(rssi):
                    rssi = self.NOT_DETECTED_VALUE
                rssi_values.append(float(rssi))

            signal = WiFiSignal(rssi_values=np.array(rssi_values))
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=floor,
                building_id=building
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from RSS-based dataset ({self.split} split)")



def RSSBased(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading RSSBased dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to RSSBasedDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = RSSBased(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = RSSBased(split='all', download=True)

        >>> # Load only training set
        >>> train = RSSBased(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = RSSBasedDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = RSSBasedDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = RSSBasedDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = RSSBasedDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return RSSBasedDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

