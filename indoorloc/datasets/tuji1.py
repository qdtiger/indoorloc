"""
TUJI1 (TUT Indoor) WiFi Dataset Implementation

Multi-device WiFi fingerprinting dataset collected at Tampere University
with high measurement density using 5 different commercial devices.

Reference:
    Klus, L., Klus, R., Lohan, E.S., et al. (2024). TUJI1 Dataset: Multi-device
    dataset for indoor localization with high measurement density.
    Data in Brief, 110356. DOI: 10.1016/j.dib.2024.110356

Dataset URL: https://zenodo.org/record/7641701
"""
import zipfile
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
class TUJI1Dataset(WiFiDataset):
    """TUJI1 (TUT Indoor) WiFi Fingerprinting Dataset.

    Multi-device WiFi dataset with high measurement density, collected using
    5 different commercial devices (Samsung S20, S7, POCO, Tab S7, A12).
    Contains over 300 distinct APs with fine-grained spatial coverage.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/tuji1).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from Zenodo
        >>> dataset = iloc.TUJI1(download=True, split='train')

    Dataset structure (after extraction):
        data_root/
        ├── RSS_training.csv
        ├── RSS_testing.csv
        ├── Coordinates_training.csv
        └── Coordinates_testing.csv
    """

    # Zenodo download URL
    ZENODO_URL = "https://zenodo.org/api/records/7641701/files/DATASET.zip/content"
    ZIP_FILENAME = "DATASET.zip"

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # File mapping
    FILE_MAPPING = {
        'train': {
            'rss': 'RSS_training.csv',
            'coords': 'Coordinates_training.csv',
        },
        'test': {
            'rss': 'RSS_testing.csv',
            'coords': 'Coordinates_testing.csv',
        },
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        floor: Union[int, List[int], str] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self._num_aps = None  # Will be determined from data
        self._ap_list = None  # List of all seen BSSIDs
        self._floor_param = floor
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
        return 'TUJI1'

    @property
    def num_aps(self) -> int:
        if self._num_aps is None:
            return 0
        return self._num_aps

    @classmethod
    def list_floors(cls, data_root: Optional[str] = None) -> List[int]:
        """List all available floors in the dataset.

        Note: TUJI1 is a single-floor dataset (floor 0).
        """
        return [0]

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        files = self.FILE_MAPPING.get(self.split)
        if files is None:
            return False
        rss_file = self.data_root / files['rss']
        coords_file = self.data_root / files['coords']
        return rss_file.exists() and coords_file.exists()

    def _download(self) -> None:
        """Download TUJI1 dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.ZIP_FILENAME

        # Download zip file
        if not zip_path.exists():
            print(f"Downloading TUJI1 dataset from Zenodo...")
            try:
                download_url(
                    url=self.ZENODO_URL,
                    root=self.data_root,
                    filename=self.ZIP_FILENAME,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download TUJI1 dataset: {e}\n"
                    f"Please download manually from: https://zenodo.org/record/7641701"
                )

        # Extract required CSV files from zip
        print(f"Extracting dataset files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for split_files in self.FILE_MAPPING.values():
                    for filename in split_files.values():
                        zip_member = f"DATASET/{filename}"
                        if zip_member in zf.namelist():
                            with zf.open(zip_member) as src:
                                target_path = self.data_root / filename
                                with open(target_path, 'wb') as dst:
                                    dst.write(src.read())
                            print(f"  Extracted: {filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")

    def _load_data(self) -> None:
        """Load TUJI1 dataset from separate RSS and coordinate CSV files."""
        files = self.FILE_MAPPING[self.split]
        rss_file = self.data_root / files['rss']
        coords_file = self.data_root / files['coords']

        if not rss_file.exists():
            raise FileNotFoundError(f"RSS file not found: {rss_file}")
        if not coords_file.exists():
            raise FileNotFoundError(f"Coordinates file not found: {coords_file}")

        # Load RSS data (no header, comma-separated RSSI values)
        rssi_data = np.loadtxt(rss_file, delimiter=',', dtype=np.float32)

        # Load coordinates (no header: x, y, floor, ?, ?, device_label)
        coords_data = np.loadtxt(coords_file, delimiter=',', dtype=np.float32)

        if len(rssi_data) != len(coords_data):
            raise ValueError(
                f"RSS and coordinate files have different lengths: "
                f"{len(rssi_data)} vs {len(coords_data)}"
            )

        # Store number of APs
        self._num_aps = rssi_data.shape[1]
        self._available_floors = [0]  # Single floor dataset

        # Filter by floor parameter (TUJI1 only has floor 0)
        if self._floor_param != 'all':
            if isinstance(self._floor_param, int):
                if self._floor_param != 0:
                    raise ValueError(f"TUJI1 only has floor 0, requested: {self._floor_param}")
            elif 0 not in self._floor_param:
                raise ValueError(f"TUJI1 only has floor 0, requested: {self._floor_param}")

        # Process each sample
        for i in range(len(rssi_data)):
            # Create WiFi signal
            signal = WiFiSignal(rssi_values=rssi_data[i])

            # Parse coordinates (x, y, floor from columns 0, 1, 2)
            x_val = float(coords_data[i, 0])
            y_val = float(coords_data[i, 1])
            floor_val = int(coords_data[i, 2]) if coords_data.shape[1] > 2 else 0

            # Create location
            location = Location(
                coordinate=Coordinate(x=x_val, y=y_val),
                floor=floor_val,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from TUJI1 dataset ({self._num_aps} APs)")



def TUJI1(data_root=None, split=None, download=False, floor='all', **kwargs):
    """
    Convenience function for loading TUJI1 dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        floor: Floor(s) to load. Can be:
            - 'all': Load all floors (default)
            - Single floor: 0, 1, 2, etc.
            - List of floors: [0, 1, 2]
        **kwargs: Additional arguments passed to TUJI1Dataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = TUJI1(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = TUJI1(split='all', download=True)

        >>> # Load specific floor(s)
        >>> train = TUJI1(floor=[0, 1], split='train')

        >>> # List available floors
        >>> TUJI1.list_floors()
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = TUJI1Dataset(
            data_root=data_root,
            split='train',
            download=download,
            floor=floor,
            **kwargs
        )
        test_dataset = TUJI1Dataset(
            data_root=data_root,
            split='test',
            download=download,
            floor=floor,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = TUJI1Dataset(
            data_root=data_root,
            split='train',
            download=download,
            floor=floor,
            **kwargs
        )
        test_dataset = TUJI1Dataset(
            data_root=data_root,
            split='test',
            download=download,
            floor=floor,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return TUJI1Dataset(
            data_root=data_root,
            split=split,
            download=download,
            floor=floor,
            **kwargs
        )


# Attach class method to convenience function
TUJI1.list_floors = TUJI1Dataset.list_floors

