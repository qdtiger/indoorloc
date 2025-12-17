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
from typing import Optional, Any, Dict, List, Union

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
            'test': 'HCXY/Testing_HCXY_All.csv',
        },
        'SYL': {
            'train': 'SYL/Training_SYL_All_30.csv',
            'test': 'SYL/Testing_SYL_All.csv',
        },
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        building: Union[str, List[str]] = 'all',
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        # Handle building parameter (支持单个、列表或 'all')
        if building == 'all':
            self._buildings = list(self.BUILDINGS.keys())
        elif isinstance(building, list):
            for b in building:
                if b not in self.BUILDINGS:
                    raise ValueError(f"Unknown building: {b}. Choose from {list(self.BUILDINGS.keys())}")
            self._buildings = building
        else:
            if building not in self.BUILDINGS:
                raise ValueError(f"Unknown building: {building}. Choose from {list(self.BUILDINGS.keys())}")
            self._buildings = [building]

        self.building = self._buildings[0]  # 兼容性
        self.building_id = self.BUILDINGS[self.building]
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

    @classmethod
    def list_buildings(cls) -> List[str]:
        """List all available buildings.

        Returns:
            List of building names: ['CETC331', 'HCXY', 'SYL']
        """
        return list(cls.BUILDINGS.keys())

    def _check_exists(self) -> bool:
        """Check if dataset files exist for all selected buildings."""
        for building in self._buildings:
            filepath = self.data_root / self.FILES[building][self.split]
            if not filepath.exists():
                return False
        return True

    def _download(self) -> None:
        """Download SODIndoorLoc dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)

        # Download files for all selected buildings
        for building in self._buildings:
            filename = self.FILES[building][self.split]
            target_path = self.data_root / filename

            if target_path.exists():
                continue

            url = f"{self.BASE_URL}/{filename}"
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
        """Load data from CSV files for all selected buildings."""
        # Pass 1: Determine max AP count across all buildings for uniform signal length
        building_specs = []
        total_aps = 0

        for building in self._buildings:
            filepath = self.data_root / self.FILES[building][self.split]
            if not filepath.exists():
                print(f"Warning: File not found for {building}: {filepath}")
                continue

            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                header = next(csv.reader(f))

            coord_start_idx = next(
                (i for i, col in enumerate(header)
                 if col in ['ECoord', 'NCoord', 'FloorID', 'BuildingID']),
                None
            )
            if coord_start_idx is None:
                raise ValueError(f"Could not find coordinate columns in CSV for {building}")

            num_aps = coord_start_idx
            total_aps = max(total_aps, num_aps)
            building_specs.append((building, filepath, coord_start_idx, num_aps))

        # Pass 2: Load data with uniform signal length (pad with NOT_DETECTED_VALUE)
        for building, filepath, coord_start_idx, num_aps in building_specs:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) < coord_start_idx + 4:
                        continue

                    # Parse RSSI values, pad to total_aps length
                    rssi_values = np.full(total_aps, self.NOT_DETECTED_VALUE, dtype=np.float32)
                    rssi_values[:num_aps] = [
                        float(row[i]) if row[i] else self.NOT_DETECTED_VALUE
                        for i in range(num_aps)
                    ]

                    signal = WiFiSignal(rssi_values=rssi_values)
                    self._signals.append(signal)

                    # Parse location
                    e_coord = float(row[coord_start_idx])
                    n_coord = float(row[coord_start_idx + 1])
                    floor = int(row[coord_start_idx + 2])
                    building_id = str(int(row[coord_start_idx + 3]))

                    coordinate = Coordinate(
                        x=e_coord,
                        y=n_coord,
                        latitude=n_coord,
                        longitude=e_coord
                    )
                    location = Location(
                        coordinate=coordinate,
                        floor=floor,
                        building_id=building_id
                    )
                    self._locations.append(location)

                    metadata = {
                        'building': building,
                        'num_detected_aps': int(np.sum(rssi_values != self.NOT_DETECTED_VALUE)),
                    }
                    self._metadata.append(metadata)

        self._num_aps = total_aps
        building_info = f" (building: {self._buildings})" if len(self._buildings) < 3 else ""
        print(f"Loaded {len(self._signals)} samples from SODIndoorLoc{building_info}")

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


def SODIndoorLoc(building='all', data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading SODIndoorLoc dataset.

    Args:
        building: Building(s) to load. Can be:
            - 'all': Load all buildings (default)
            - Single building: 'CETC331', 'HCXY', 'SYL'
            - List of buildings: ['CETC331', 'HCXY']
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to SODIndoorLocDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load all buildings
        >>> train, test = SODIndoorLoc(download=True)

        >>> # Load specific building(s)
        >>> train = SODIndoorLoc(building=['CETC331', 'HCXY'], split='train')

        >>> # List available buildings
        >>> SODIndoorLoc.list_buildings()
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


# Attach class method to convenience function
SODIndoorLoc.list_buildings = SODIndoorLocDataset.list_buildings
