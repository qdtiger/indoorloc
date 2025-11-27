"""
UJIndoorLoc Dataset Implementation

The UJIndoorLoc dataset is a popular WiFi fingerprinting dataset collected at
Universitat Jaume I, Spain. It covers 3 buildings with 4-5 floors each.

Reference:
    Torres-Sospedra, J., et al. (2014). UJIndoorLoc: A new multi-building and
    multi-floor database for WLAN fingerprint-based indoor localization problems.
    In 2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN).

Dataset URL: https://archive.ics.uci.edu/dataset/310/ujiindoorloc
"""
import csv
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_and_extract_zip


@DATASETS.register_module()
class UJIndoorLocDataset(WiFiDataset):
    """UJIndoorLoc WiFi Fingerprinting Dataset.

    This dataset contains WiFi RSSI fingerprints from 520 access points
    collected in 3 buildings at Universitat Jaume I.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ujindoorloc).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Auto-download to default cache directory
        >>> dataset = iloc.UJIndoorLoc(download=True)
        >>> # Or specify custom directory
        >>> dataset = iloc.UJIndoorLoc(data_root='./data', download=True)

    Dataset structure:
        data_root/
        ├── trainingData.csv
        └── validationData.csv

    CSV columns (0-indexed):
        0-519: WAP001-WAP520 (RSSI values, 100 = not detected)
        520: LONGITUDE
        521: LATITUDE
        522: FLOOR
        523: BUILDINGID
        524: SPACEID
        525: RELATIVEPOSITION
        526: USERID
        527: PHONEID
        528: TIMESTAMP
    """

    # Dataset constants
    NUM_WAPS = 520
    NOT_DETECTED_VALUE = 100

    # Download URLs (primary and backup mirrors)
    DOWNLOAD_URLS = [
        'https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip',
    ]

    # Required files
    REQUIRED_FILES = ['trainingData.csv', 'validationData.csv']

    # Column indices
    COL_LONGITUDE = 520
    COL_LATITUDE = 521
    COL_FLOOR = 522
    COL_BUILDING = 523
    COL_SPACEID = 524
    COL_RELATIVEPOSITION = 525
    COL_USERID = 526
    COL_PHONEID = 527
    COL_TIMESTAMP = 528

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
        # Store kwargs before calling super().__init__
        self._kwargs = kwargs
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
        return 'UJIndoorLoc'

    @property
    def num_aps(self) -> int:
        return self.NUM_WAPS

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download UJIndoorLoc dataset from UCI repository."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        # Try each mirror until one succeeds
        last_error = None
        for url in self.DOWNLOAD_URLS:
            try:
                download_and_extract_zip(
                    url=url,
                    root=self.data_root,
                    extract_files=self.REQUIRED_FILES,
                )
                return
            except Exception as e:
                last_error = e
                print(f"Mirror {url} failed: {e}")
                continue

        raise RuntimeError(
            f"Failed to download dataset from all mirrors.\n"
            f"Last error: {last_error}\n"
            f"Please download manually from:\n"
            f"https://archive.ics.uci.edu/dataset/310/ujiindoorloc"
        )

    def _load_data(self) -> None:
        """Load data from CSV files."""
        # Determine file to load
        if self.split == 'train':
            filename = 'trainingData.csv'
        elif self.split in ('test', 'val', 'validation'):
            filename = 'validationData.csv'
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train' or 'test'.")

        filepath = self.data_root / filename

        # Load CSV data
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 529:
                    continue  # Skip incomplete rows

                # Parse RSSI values (columns 0-519)
                rssi_values = np.array(
                    [float(row[i]) for i in range(self.NUM_WAPS)],
                    dtype=np.float32
                )

                # Create WiFi signal
                signal = WiFiSignal(rssi_values=rssi_values)
                self._signals.append(signal)

                # Parse location
                longitude = float(row[self.COL_LONGITUDE])
                latitude = float(row[self.COL_LATITUDE])
                floor = int(row[self.COL_FLOOR])
                building_id = str(int(row[self.COL_BUILDING]))

                # UJIndoorLoc uses UTM coordinates (longitude/latitude are actually x/y in meters)
                # The coordinates are already in a local reference frame
                coordinate = Coordinate(
                    x=longitude,  # In UJIndoorLoc, LONGITUDE column is actually x coordinate
                    y=latitude,   # LATITUDE column is actually y coordinate
                    latitude=latitude,
                    longitude=longitude
                )

                location = Location(
                    coordinate=coordinate,
                    floor=floor,
                    building_id=building_id
                )
                self._locations.append(location)

                # Store metadata
                metadata = {
                    'space_id': int(row[self.COL_SPACEID]),
                    'relative_position': int(row[self.COL_RELATIVEPOSITION]),
                    'user_id': int(row[self.COL_USERID]),
                    'phone_id': int(row[self.COL_PHONEID]),
                    'timestamp': float(row[self.COL_TIMESTAMP]) if row[self.COL_TIMESTAMP] else None
                }
                self._metadata.append(metadata)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute UJIndoorLoc-specific statistics."""
        stats = super().get_statistics()

        # User and phone statistics
        user_ids = [m['user_id'] for m in self._metadata]
        phone_ids = [m['phone_id'] for m in self._metadata]

        stats['num_users'] = len(set(user_ids))
        stats['num_phones'] = len(set(phone_ids))

        # Building-floor distribution
        building_floor_counts = {}
        for loc in self._locations:
            key = (loc.building_id, loc.floor)
            building_floor_counts[key] = building_floor_counts.get(key, 0) + 1

        stats['samples_per_building_floor'] = building_floor_counts

        return stats



def UJIndoorLoc(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading UJIndoorLoc dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to UJIndoorLocDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = UJIndoorLoc(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = UJIndoorLoc(split='all', download=True)

        >>> # Load only training set
        >>> train = UJIndoorLoc(split='train', download=True)

        >>> # Load only test set
        >>> test = UJIndoorLoc(split='test', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = UJIndoorLocDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = UJIndoorLocDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = UJIndoorLocDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = UJIndoorLocDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return UJIndoorLocDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )
