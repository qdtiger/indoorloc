"""
Tampere University WiFi Dataset Implementation

Large-scale WiFi fingerprinting dataset collected at Tampere University
covering multiple buildings and floors with high spatial resolution.

Reference:
    Lohan, E. S., Torres-Sospedra, J., et al. (2017). Wi-Fi Crowdsourced
    Fingerprinting Dataset for Indoor Positioning. Data, 2(4), 32.
    DOI: 10.3390/data2040032

Dataset URL: https://zenodo.org/record/889798
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
class TampereDataset(WiFiDataset):
    """Tampere University WiFi Fingerprinting Dataset.

    Large-scale WiFi dataset with crowdsourced fingerprints collected
    at Tampere University. Contains 687 training and 3951 test fingerprints
    collected with 21 devices in a 4-floor building.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/tampere).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from Zenodo
        >>> dataset = iloc.Tampere(download=True, split='train')

    Dataset structure (after extraction):
        data_root/
        ├── Training_rss_21Aug17.csv      # RSSI values (no header)
        ├── Training_coordinates_21Aug17.csv  # x, y, floor (no header)
        ├── Test_rss_21Aug17.csv
        └── Test_coordinates_21Aug17.csv
    """

    # Zenodo download URL
    ZENODO_URL = "https://zenodo.org/api/records/889798/files/DISTRIBUTED_OPENSOURCE.zip/content"
    ZIP_FILENAME = "DISTRIBUTED_OPENSOURCE.zip"

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # File mapping for RSS and coordinates (inside FINGERPRINTING_DB folder)
    FILE_MAPPING = {
        'train': {
            'rss': 'Training_rss_21Aug17.csv',
            'coords': 'Training_coordinates_21Aug17.csv',
        },
        'test': {
            'rss': 'Test_rss_21Aug17.csv',
            'coords': 'Test_coordinates_21Aug17.csv',
        },
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        building: Union[str, List[str]] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self._num_aps = None  # Will be determined from data
        self._building_param = building
        self._available_buildings: List[str] = []

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
        return 'Tampere'

    @property
    def num_aps(self) -> int:
        if self._num_aps is None:
            return 0
        return self._num_aps

    @classmethod
    def list_buildings(cls, data_root: Optional[str] = None) -> List[str]:
        """List all available buildings in the dataset.

        Note: Tampere dataset is from a single building with 4 floors.
        """
        return ['1']  # Single building dataset

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        files = self.FILE_MAPPING.get(self.split)
        if files is None:
            return False
        rss_file = self.data_root / files['rss']
        coords_file = self.data_root / files['coords']
        return rss_file.exists() and coords_file.exists()

    def _download(self) -> None:
        """Download Tampere dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.ZIP_FILENAME

        # Download zip file
        if not zip_path.exists():
            print(f"Downloading Tampere dataset from Zenodo...")
            try:
                download_url(
                    url=self.ZENODO_URL,
                    root=self.data_root,
                    filename=self.ZIP_FILENAME,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Tampere dataset: {e}\n"
                    f"Please download manually from: https://zenodo.org/record/889798"
                )

        # Extract required CSV files from zip
        print(f"Extracting dataset files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for split_files in self.FILE_MAPPING.values():
                    for filename in split_files.values():
                        zip_member = f"FINGERPRINTING_DB/{filename}"
                        if zip_member in zf.namelist():
                            # Extract to data_root directly (flatten structure)
                            with zf.open(zip_member) as src:
                                target_path = self.data_root / filename
                                with open(target_path, 'wb') as dst:
                                    dst.write(src.read())
                            print(f"  Extracted: {filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")

    def _load_data(self) -> None:
        """Load Tampere dataset from separate RSS and coordinate CSV files."""
        files = self.FILE_MAPPING[self.split]
        rss_file = self.data_root / files['rss']
        coords_file = self.data_root / files['coords']

        if not rss_file.exists():
            raise FileNotFoundError(f"RSS file not found: {rss_file}")
        if not coords_file.exists():
            raise FileNotFoundError(f"Coordinates file not found: {coords_file}")

        # Load RSS data (no header, comma-separated RSSI values)
        rssi_data = np.loadtxt(rss_file, delimiter=',', dtype=np.float32)

        # Load coordinates (no header: x, y, floor)
        coords_data = np.loadtxt(coords_file, delimiter=',', dtype=np.float32)

        if len(rssi_data) != len(coords_data):
            raise ValueError(
                f"RSS and coordinate files have different lengths: "
                f"{len(rssi_data)} vs {len(coords_data)}"
            )

        # Store number of APs
        self._num_aps = rssi_data.shape[1]
        self._available_buildings = ['1']  # Single building

        # Process each sample
        for i in range(len(rssi_data)):
            # Create WiFi signal
            signal = WiFiSignal(rssi_values=rssi_data[i])

            # Parse coordinates (x, y, floor)
            x_val = float(coords_data[i, 0])
            y_val = float(coords_data[i, 1])
            floor_val = int(coords_data[i, 2]) if coords_data.shape[1] > 2 else 0

            # Create location
            location = Location(
                coordinate=Coordinate(x=x_val, y=y_val),
                floor=floor_val,
                building_id='1'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from Tampere dataset ({self._num_aps} APs)")



def Tampere(data_root=None, split=None, download=False, building='all', **kwargs):
    """
    Convenience function for loading Tampere dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        building: Building(s) to load. Can be:
            - 'all': Load all buildings (default)
            - Single building: '1', '2', etc.
            - List of buildings: ['1', '2', '3']
        **kwargs: Additional arguments passed to TampereDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = Tampere(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = Tampere(split='all', download=True)

        >>> # Load specific building(s)
        >>> train = Tampere(building=['1', '2'], split='train')

        >>> # List available buildings
        >>> Tampere.list_buildings()
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = TampereDataset(
            data_root=data_root,
            split='train',
            download=download,
            building=building,
            **kwargs
        )
        test_dataset = TampereDataset(
            data_root=data_root,
            split='test',
            download=download,
            building=building,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = TampereDataset(
            data_root=data_root,
            split='train',
            download=download,
            building=building,
            **kwargs
        )
        test_dataset = TampereDataset(
            data_root=data_root,
            split='test',
            download=download,
            building=building,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return TampereDataset(
            data_root=data_root,
            split=split,
            download=download,
            building=building,
            **kwargs
        )


# Attach class method to convenience function
Tampere.list_buildings = TampereDataset.list_buildings

