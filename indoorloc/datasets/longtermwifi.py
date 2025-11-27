"""
Long-Term WiFi Fingerprinting Dataset Implementation

A multi-building WiFi fingerprinting dataset collected over a long period
to study temporal variations in WiFi fingerprints.

Reference:
    Mendoza-Silva, G. M., et al. (2016). Long-term WiFi fingerprinting dataset
    for research on robust indoor positioning.
    Data, 1(1), 3.

Dataset URL: https://zenodo.org/record/889798
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np
import pandas as pd

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class LongTermWiFiDataset(WiFiDataset):
    """Long-Term WiFi Fingerprinting Dataset.

    Multi-building WiFi fingerprinting dataset with temporal variations,
    collected over multiple months to study long-term stability.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/longtermwifi).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        building: Building identifier (0, 1, or 2). None for all buildings.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from Zenodo
        >>> dataset = iloc.LongTermWiFi(download=True)
        >>> # Specific building
        >>> dataset = iloc.LongTermWiFi(building=0, download=True)

    Dataset structure:
        data_root/
        ├── TRNDB0.txt   # Training Building 0
        ├── TSTDB0.txt   # Testing Building 0
        ├── TRNDB1.txt   # Training Building 1
        ├── TSTDB1.txt   # Testing Building 1
        ├── TRNDB2.txt   # Training Building 2
        └── TSTDB2.txt   # Testing Building 2
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '889798'

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # File naming pattern
    FILE_PATTERNS = {
        'train': 'TRNDB{building}.txt',
        'test': 'TSTDB{building}.txt',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        building: Optional[int] = None,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self.building = building
        self._num_aps = None  # Will be determined from data

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
        return 'LongTermWiFi'

    @property
    def num_aps(self) -> int:
        if self._num_aps is None:
            return 0
        return self._num_aps

    def _get_required_files(self) -> List[str]:
        """Get list of required files based on building selection."""
        if self.building is not None:
            # Single building
            pattern = self.FILE_PATTERNS[self.split]
            return [pattern.format(building=self.building)]
        else:
            # All buildings
            files = []
            for building_id in [0, 1, 2]:
                for split in ['train', 'test']:
                    pattern = self.FILE_PATTERNS[split]
                    files.append(pattern.format(building=building_id))
            return files

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        required_files = self._get_required_files()
        return all(
            (self.data_root / f).exists()
            for f in required_files
        )

    def _download(self) -> None:
        """Download Long-Term WiFi dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading Long-Term WiFi dataset from Zenodo...")

        try:
            # Download all files from Zenodo record
            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=None,  # Download all files
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Long-Term WiFi dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load Long-Term WiFi dataset from files."""
        if self.building is not None:
            # Load single building
            buildings_to_load = [self.building]
        else:
            # Load all buildings
            buildings_to_load = [0, 1, 2]

        all_signals = []
        all_locations = []

        for building_id in buildings_to_load:
            pattern = self.FILE_PATTERNS[self.split]
            filename = pattern.format(building=building_id)
            filepath = self.data_root / filename

            if not filepath.exists():
                continue

            # Load data file
            # Format: each line is space-separated values
            # Last 3 columns: X Y FLOOR
            # Other columns: RSSI values for APs
            try:
                data = np.loadtxt(filepath, dtype=float)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue

            if len(data) == 0:
                continue

            # Extract coordinates and RSSI values
            # Last 3 columns are X, Y, FLOOR
            rssi_data = data[:, :-3]
            coordinates = data[:, -3:]

            # Store number of APs
            if self._num_aps is None:
                self._num_aps = rssi_data.shape[1]

            # Process each sample
            for i in range(len(data)):
                # Create WiFi signal
                rssi_values = rssi_data[i].astype(np.float32)
                signal = WiFiSignal(rssi_values=rssi_values)

                # Create location
                x, y, floor = coordinates[i]
                location = Location(
                    coordinate=Coordinate(x=x, y=y),
                    floor=int(floor),
                    building_id=str(building_id)
                )

                all_signals.append(signal)
                all_locations.append(location)

        # Store data
        self._signals = all_signals
        self._locations = all_locations

        if len(self._signals) == 0:
            raise RuntimeError(
                f"No data loaded for split='{self.split}', building={self.building}"
            )

        print(f"Loaded {len(self._signals)} samples from Long-Term WiFi dataset")



def LongTermWiFi(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading LongTermWiFi dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to LongTermWiFiDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = LongTermWiFi(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = LongTermWiFi(split='all', download=True)

        >>> # Load only training set
        >>> train = LongTermWiFi(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return LongTermWiFiDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

