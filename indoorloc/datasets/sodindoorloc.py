"""
SODIndoorLoc Dataset Implementation

SODIndoorLoc is a supplementary open dataset for WiFi indoor localization
based on received signal strength. It covers 3 buildings across multiple
floors with about 8000 square meters.

Reference:
    Ren, W., et al. (2022). Supplementary open dataset for WiFi indoor
    localization based on received signal strength. Satellite Navigation.

Dataset URL: https://github.com/renwudao24/SODIndoorLoc
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
from ..utils.download import download_url


@DATASETS.register_module()
class SODIndoorLocDataset(WiFiDataset):
    """SODIndoorLoc WiFi Fingerprinting Dataset.

    This dataset contains WiFi RSSI fingerprints from 3 buildings
    (CETC331, HCXY, SYL) with about 8000 square meters coverage.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/sodindoorloc).
        building: Building name ('CETC331', 'HCXY', or 'SYL').
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Auto-download CETC331 building data
        >>> dataset = iloc.SODIndoorLoc(building='CETC331', download=True)
        >>> # Load HCXY building test data
        >>> test_data = iloc.SODIndoorLoc(building='HCXY', split='test', download=True)

    Dataset structure:
        data_root/
        ├── Training_CETC331.csv
        ├── Testing_CETC331.csv
        ├── Training_HCXY_All_30.csv
        ├── Training_HCXY_AP_30.csv
        └── ...

    CSV columns:
        - MAC1, MAC2, ... : RSSI values (100 = not detected)
        - ECoord: East coordinate in meters
        - NCoord: North coordinate in meters
        - FloorID: Floor level (1-4)
        - BuildingID: Building ID (1-3)
    """

    # Building info
    BUILDINGS = {
        'CETC331': 1,
        'HCXY': 2,
        'SYL': 3,
    }

    NOT_DETECTED_VALUE = 100

    # Download URLs for CSV files
    BASE_URL = "https://raw.githubusercontent.com/renwudao24/SODIndoorLoc/main"

    FILES = {
        'CETC331': {
            'train': 'CETC331/Training_CETC331.csv',
            'test': 'CETC331/Testing_CETC331.csv',
        },
        'HCXY': {
            'train': 'HCXY/Training_HCXY_All_30.csv',
            'test': 'HCXY/Testing_HCXY.csv',
        },
        'SYL': {
            'train': 'SYL/Training_SYL_All_30.csv',
            'test': 'SYL/Testing_SYL.csv',
        },
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        building: str = 'CETC331',
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        if building not in self.BUILDINGS:
            raise ValueError(
                f"Unknown building: {building}. "
                f"Choose from {list(self.BUILDINGS.keys())}"
            )

        self.building = building
        self.building_id = self.BUILDINGS[building]
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
        return 'SODIndoorLoc'

    @property
    def num_aps(self) -> int:
        return self._num_aps if self._num_aps else 0

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filepath = self.data_root / self.FILES[self.building][self.split]
        return filepath.exists()

    def _download(self) -> None:
        """Download SODIndoorLoc dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)

        # Download the specific file
        filename = self.FILES[self.building][self.split]
        url = f"{self.BASE_URL}/{filename}"

        target_path = self.data_root / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {filename} from GitHub...")
        try:
            download_url(
                url=url,
                root=target_path.parent,
                filename=target_path.name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset.\n"
                f"Error: {e}\n"
                f"Please download manually from:\n"
                f"https://github.com/renwudao24/SODIndoorLoc"
            )

    def _load_data(self) -> None:
        """Load data from CSV files."""
        filepath = self.data_root / self.FILES[self.building][self.split]

        # Load CSV data
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header to get MAC addresses

            # Identify column indices
            # Find where coordinate columns start
            coord_start_idx = None
            for i, col in enumerate(header):
                if col in ['ECoord', 'NCoord', 'FloorID', 'BuildingID']:
                    coord_start_idx = i
                    break

            if coord_start_idx is None:
                raise ValueError("Could not find coordinate columns in CSV")

            # Number of APs = columns before coordinates
            self._num_aps = coord_start_idx
            mac_addresses = header[:self._num_aps]

            for row in reader:
                if len(row) < coord_start_idx + 4:
                    continue  # Skip incomplete rows

                # Parse RSSI values
                rssi_values = np.array(
                    [float(row[i]) if row[i] else 100.0
                     for i in range(self._num_aps)],
                    dtype=np.float32
                )

                # Create WiFi signal
                signal = WiFiSignal(rssi_values=rssi_values)
                self._signals.append(signal)

                # Parse location
                e_coord = float(row[coord_start_idx])
                n_coord = float(row[coord_start_idx + 1])
                floor = int(row[coord_start_idx + 2])
                building_id = str(int(row[coord_start_idx + 3]))

                # SODIndoorLoc uses East/North coordinates in meters
                coordinate = Coordinate(
                    x=e_coord,
                    y=n_coord,
                    latitude=n_coord,  # Use north as latitude
                    longitude=e_coord   # Use east as longitude
                )

                location = Location(
                    coordinate=coordinate,
                    floor=floor,
                    building_id=building_id
                )
                self._locations.append(location)

                # Store metadata
                metadata = {
                    'building': self.building,
                    'num_detected_aps': np.sum(rssi_values != self.NOT_DETECTED_VALUE),
                }
                self._metadata.append(metadata)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute SODIndoorLoc-specific statistics."""
        stats = super().get_statistics()
        stats['building'] = self.building

        # AP detection statistics
        detection_counts = [m['num_detected_aps'] for m in self._metadata]
        stats['avg_detected_aps'] = np.mean(detection_counts)
        stats['min_detected_aps'] = min(detection_counts)
        stats['max_detected_aps'] = max(detection_counts)

        return stats


def SODIndoorLoc(building='CETC331', data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading SODIndoorLoc dataset.

    Args:
        building: Building name ('CETC331', 'HCXY', or 'SYL')
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to SODIndoorLocDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = SODIndoorLoc(building='CETC331', download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = SODIndoorLoc(building='CETC331', split='all', download=True)

        >>> # Load only training set
        >>> train = SODIndoorLoc(building='HCXY', split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = SODIndoorLocDataset(
            building=building,
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = SODIndoorLocDataset(
            building=building,
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = SODIndoorLocDataset(
            building=building,
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = SODIndoorLocDataset(
            building=building,
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return SODIndoorLocDataset(
            building=building,
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )
