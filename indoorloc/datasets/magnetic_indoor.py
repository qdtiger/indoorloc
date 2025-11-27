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
from typing import Optional, Any, Dict, List
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
        transform: Optional[Any] = None,
        normalize: bool = False,
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
        return 'MagneticIndoor'

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

        print(f"Loaded {len(self._signals)} samples from magnetic field dataset ({self.split} split)")



def MagneticIndoor(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading MagneticIndoor dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to MagneticIndoorDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = MagneticIndoor(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = MagneticIndoor(split='all', download=True)

        >>> # Load only training set
        >>> train = MagneticIndoor(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
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
            **kwargs
        )
        test_dataset = MagneticIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return MagneticIndoorDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

