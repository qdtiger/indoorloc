"""
Magnetic Field Indoor Localization Dataset Implementation

Geomagnetic field-based indoor positioning dataset using magnetometer
measurements for location fingerprinting.

Reference:
    Indoor Localization Using Geomagnetic Field Fingerprinting.
    Zenodo Repository.
    https://zenodo.org/record/4321098

Dataset URL: https://zenodo.org/record/4321098
"""
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import numpy as np

from .base import MagneticDataset
from ..signals.magnetometer import MagnetometerSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class MagneticIndoorDataset(MagneticDataset):
    """Magnetic Field Indoor Localization Dataset.

    Indoor positioning dataset using geomagnetic field measurements
    for location fingerprinting and navigation.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/magnetic_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download magnetic field dataset
        >>> dataset = iloc.MagneticIndoor(download=True, split='train')
        >>> # Access magnetic field signal
        >>> signal = dataset[0]
        >>> print(f"Magnetic field: {signal.magnetic_field}")

    Dataset structure:
        data_root/
        ├── train_magnetic.csv
        └── test_magnetic.csv

    CSV format:
        - Columns: timestamp, x, y, floor, building, mag_x, mag_y, mag_z, mag_total
        - Magnetic field values in μT (microtesla)
        - Multiple readings per location for robustness
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '4321098'

    # Dataset constants
    MIN_FIELD = -100.0  # μT
    MAX_FIELD = 100.0   # μT

    # File mapping
    FILE_MAPPING = {
        'train': 'train_magnetic.csv',
        'test': 'test_magnetic.csv',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        building: Union[str, List[str]] = 'all',
        floor: Union[int, List[int], str] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = False,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self._building_param = building
        self._floor_param = floor
        self._available_buildings: List[str] = []
        self._available_floors: List[int] = []

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
        return 'MagneticIndoor'

    @classmethod
    def list_buildings(cls, data_root: Optional[str] = None) -> List[str]:
        """List all available buildings in the dataset.

        Args:
            data_root: Root directory containing the dataset files.

        Returns:
            List of building IDs found in the dataset.
        """
        from ..utils.download import get_data_home

        if data_root is None:
            root = get_data_home() / 'magnetic_indoor'
        else:
            root = Path(data_root)

        train_file = root / 'train_magnetic.csv'
        if not train_file.exists():
            return []

        try:
            import pandas as pd
            df = pd.read_csv(train_file, usecols=['building'])
            return sorted(df['building'].astype(str).unique().tolist())
        except Exception:
            return []

    @classmethod
    def list_floors(cls, data_root: Optional[str] = None) -> List[int]:
        """List all available floors in the dataset.

        Args:
            data_root: Root directory containing the dataset files.

        Returns:
            List of floor numbers found in the dataset.
        """
        from ..utils.download import get_data_home

        if data_root is None:
            root = get_data_home() / 'magnetic_indoor'
        else:
            root = Path(data_root)

        train_file = root / 'train_magnetic.csv'
        if not train_file.exists():
            return []

        try:
            import pandas as pd
            df = pd.read_csv(train_file, usecols=['floor'])
            return sorted(df['floor'].astype(int).unique().tolist())
        except Exception:
            return []

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        return (self.data_root / filename).exists()

    def _download(self) -> None:
        """Download magnetic field dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading magnetic field indoor dataset from Zenodo...")

        try:
            # Download all required files
            all_files = list(self.FILE_MAPPING.values())
            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=all_files,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download magnetic field dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load magnetic field dataset from CSV file."""
        filename = self.FILE_MAPPING[self.split]
        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load magnetic field dataset.\n"
                "Install with: pip install pandas"
            )

        # Load CSV
        df = pd.read_csv(filepath)

        # Store available buildings and floors
        if 'building' in df.columns:
            self._available_buildings = sorted(df['building'].astype(str).unique().tolist())
        if 'floor' in df.columns:
            self._available_floors = sorted(df['floor'].astype(int).unique().tolist())

        # Filter by building
        if self._building_param != 'all' and 'building' in df.columns:
            if isinstance(self._building_param, list):
                selected_buildings = [str(b) for b in self._building_param]
            else:
                selected_buildings = [str(self._building_param)]
            df = df[df['building'].astype(str).isin(selected_buildings)]

        # Filter by floor
        if self._floor_param != 'all' and 'floor' in df.columns:
            if isinstance(self._floor_param, int):
                selected_floors = [self._floor_param]
            else:
                selected_floors = list(self._floor_param)
            df = df[df['floor'].astype(int).isin(selected_floors)]

        if len(df) == 0:
            raise ValueError(f"No data found for building={self._building_param}, floor={self._floor_param}")

        # Expected columns: timestamp, x, y, floor, building, mag_x, mag_y, mag_z, mag_total
        # Process each sample
        for idx, row in df.iterrows():
            # Extract location
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            floor = int(row.get('floor', 0))
            building = str(row.get('building', '0'))

            # Extract magnetic field measurements
            mag_x = float(row.get('mag_x', 0.0))
            mag_y = float(row.get('mag_y', 0.0))
            mag_z = float(row.get('mag_z', 0.0))

            # Create magnetic field vector
            magnetic_field = np.array([mag_x, mag_y, mag_z])

            # Create magnetometer signal
            signal = MagnetometerSignal(magnetic_field=magnetic_field)

            # Create location
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=floor,
                building_id=building
            )

            self._signals.append(signal)
            self._locations.append(location)

        filter_info = ""
        if self._building_param != 'all':
            filter_info += f" (building: {self._building_param})"
        if self._floor_param != 'all':
            filter_info += f" (floor: {self._floor_param})"
        print(f"Loaded {len(self._signals)} samples from magnetic field dataset{filter_info}")



def MagneticIndoor(data_root=None, split=None, download=False, building='all', floor='all', **kwargs):
    """
    Convenience function for loading MagneticIndoor dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        building: Building(s) to load. Can be:
            - 'all': Load all buildings (default)
            - Single building: '0', '1', etc.
            - List of buildings: ['0', '1']
        floor: Floor(s) to load. Can be:
            - 'all': Load all floors (default)
            - Single floor: 0, 1, 2, etc.
            - List of floors: [0, 1, 2]
        **kwargs: Additional arguments passed to MagneticIndoorDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = MagneticIndoor(download=True)

        >>> # Load specific building/floor
        >>> train = MagneticIndoor(building='0', floor=[1, 2], split='train')

        >>> # List available buildings/floors
        >>> MagneticIndoor.list_buildings()
        >>> MagneticIndoor.list_floors()
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            building=building,
            floor=floor,
            **kwargs
        )
        test_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            building=building,
            floor=floor,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            building=building,
            floor=floor,
            **kwargs
        )
        test_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            building=building,
            floor=floor,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return MagneticIndoorDataset(
            data_root=data_root,
            split=split,
            download=download,
            building=building,
            floor=floor,
            **kwargs
        )


# Attach class methods to convenience function
MagneticIndoor.list_buildings = MagneticIndoorDataset.list_buildings
MagneticIndoor.list_floors = MagneticIndoorDataset.list_floors

