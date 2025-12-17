"""
Long-Term WiFi Fingerprinting Dataset Implementation

WiFi fingerprinting dataset collected over 25 months at UJI Library
to study temporal variations in WiFi fingerprints.

Reference:
    Mendoza-Silva, G.M., Richter, P., Torres-Sospedra, J., Lohan, E.S., Huerta, J. (2018).
    Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning.
    Data, 3(1), 3. DOI: 10.3390/data3010003

Dataset URL: https://zenodo.org/record/1309317
"""
import zipfile
from pathlib import Path
from typing import Optional, Any, List, Union
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class LongTermWiFiDataset(WiFiDataset):
    """Long-Term WiFi Fingerprinting Dataset (UJI Library).

    WiFi fingerprinting dataset collected over 25 months at two floors
    of Universitat Jaume I library. Contains 103,584 WiFi fingerprints
    for studying long-term signal variations.

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        month: Month(s) to load (1-25, 'all', or list like [1, 5, 10]).
        floor: Floor(s) to load (3 or 4, 'all', or list).
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method.

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.LongTermWiFi(download=True)
        >>> # Specific month
        >>> train = iloc.LongTermWiFi(month=1, split='train')

    Dataset structure (after extraction):
        data_root/
        └── db/
            ├── 01/  # Month 01
            │   ├── trn01rss.csv  # Training RSS
            │   ├── trn01crd.csv  # Training coordinates
            │   ├── tst01rss.csv  # Test set 1
            │   └── ...
            ├── 02/  # Month 02
            └── ...
    """

    ZENODO_URL = "https://zenodo.org/api/records/1309317/files/UJI_LIB_DB_v2.zip/content"
    ZIP_FILENAME = "UJI_LIB_DB_v2.zip"

    NOT_DETECTED_VALUE = 100
    AVAILABLE_MONTHS = list(range(1, 26))  # 1-25
    AVAILABLE_FLOORS = [3, 5]  # Two library floors (3rd and 5th)

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        month: Union[int, List[int], str] = 'all',
        floor: Union[int, List[int], str] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        # Handle month parameter
        if month == 'all':
            self._months = self.AVAILABLE_MONTHS.copy()
        elif isinstance(month, int):
            self._months = [month]
        else:
            self._months = list(month)

        # Handle floor parameter
        if floor == 'all':
            self._floors = self.AVAILABLE_FLOORS.copy()
        elif isinstance(floor, int):
            self._floors = [floor]
        else:
            self._floors = list(floor)

        self._num_aps = None

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
        return self._num_aps or 0

    @classmethod
    def list_months(cls) -> List[int]:
        """List all available months (1-25)."""
        return cls.AVAILABLE_MONTHS.copy()

    @classmethod
    def list_floors(cls) -> List[int]:
        """List all available floors (3, 4)."""
        return cls.AVAILABLE_FLOORS.copy()

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        db_dir = self.data_root / 'db' / '01'
        return db_dir.exists() and (db_dir / 'trn01rss.csv').exists()

    def _download(self) -> None:
        """Download Long-Term WiFi dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.ZIP_FILENAME

        # Download zip file
        if not zip_path.exists():
            print("Downloading Long-Term WiFi dataset from Zenodo...")
            try:
                download_url(
                    url=self.ZENODO_URL,
                    root=self.data_root,
                    filename=self.ZIP_FILENAME,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Long-Term WiFi dataset: {e}\n"
                    f"Please download manually from: https://zenodo.org/record/1309317"
                )

        # Extract zip file
        print("Extracting dataset files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Extract only db/ folder contents
                for member in zf.namelist():
                    if member.startswith('db/') and member.endswith('.csv'):
                        zf.extract(member, self.data_root)
            print(f"Extracted to {self.data_root / 'db'}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")

    def _load_data(self) -> None:
        """Load Long-Term WiFi dataset from CSV files."""
        db_dir = self.data_root / 'db'

        # Pass 1: Determine max AP count across all files
        max_aps = 0
        file_specs = []  # (month, rss_path, crd_path)

        for month in self._months:
            month_dir = db_dir / f'{month:02d}'
            if not month_dir.exists():
                continue

            if self.split == 'train':
                # Training: load ALL trn*rss.csv files from this month
                for rss_file in sorted(month_dir.glob('trn*rss.csv')):
                    crd_file = month_dir / rss_file.name.replace('rss.csv', 'crd.csv')
                    if crd_file.exists():
                        sample_line = rss_file.read_text().split('\n')[0]
                        num_aps = len(sample_line.split(','))
                        max_aps = max(max_aps, num_aps)
                        file_specs.append((month, rss_file, crd_file))
            else:
                # Test: load ALL tst*rss.csv files from this month
                for rss_file in sorted(month_dir.glob('tst*rss.csv')):
                    crd_file = month_dir / rss_file.name.replace('rss.csv', 'crd.csv')
                    if crd_file.exists():
                        sample_line = rss_file.read_text().split('\n')[0]
                        num_aps = len(sample_line.split(','))
                        max_aps = max(max_aps, num_aps)
                        file_specs.append((month, rss_file, crd_file))

        if not file_specs:
            raise FileNotFoundError(
                f"No data files found for split='{self.split}', month={self._months}"
            )

        self._num_aps = max_aps

        # Pass 2: Load data with uniform signal length
        for month, rss_path, crd_path in file_specs:
            rss_data = np.loadtxt(rss_path, delimiter=',', dtype=np.float32)
            crd_data = np.loadtxt(crd_path, delimiter=',', dtype=np.float32)

            if len(rss_data.shape) == 1:
                rss_data = rss_data.reshape(1, -1)
            if len(crd_data.shape) == 1:
                crd_data = crd_data.reshape(1, -1)

            for i in range(len(rss_data)):
                # Pad RSSI to uniform length
                rssi_values = np.full(max_aps, self.NOT_DETECTED_VALUE, dtype=np.float32)
                rssi_values[:rss_data.shape[1]] = rss_data[i]

                # Parse coordinates (x, y, floor)
                x, y = crd_data[i, 0], crd_data[i, 1]
                floor_val = int(crd_data[i, 2]) if crd_data.shape[1] > 2 else 3

                # Filter by floor
                if floor_val not in self._floors:
                    continue

                signal = WiFiSignal(rssi_values=rssi_values)
                location = Location(
                    coordinate=Coordinate(x=float(x), y=float(y)),
                    floor=floor_val,
                    building_id='0'
                )

                self._signals.append(signal)
                self._locations.append(location)

        if not self._signals:
            raise RuntimeError(f"No samples found for floor={self._floors}")

        month_info = f" (month: {self._months})" if len(self._months) < 25 else ""
        print(f"Loaded {len(self._signals)} samples from Long-Term WiFi dataset{month_info}")



def LongTermWiFi(data_root=None, split=None, download=False, month='all', floor='all', **kwargs):
    """
    Convenience function for loading LongTermWiFi dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        month: Month(s) to load. Can be:
            - 'all': Load all 25 months (default)
            - Single month: 1, 5, 10, etc.
            - List of months: [1, 5, 10]
        floor: Floor(s) to load (3 or 4, 'all')
        **kwargs: Additional arguments passed to LongTermWiFiDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> train, test = LongTermWiFi(download=True)
        >>> train = LongTermWiFi(month=[1, 5], split='train')
        >>> LongTermWiFi.list_months()
    """
    if split is None:
        train_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='train',
            download=download,
            month=month,
            floor=floor,
            **kwargs
        )
        test_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='test',
            download=download,
            month=month,
            floor=floor,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='train',
            download=download,
            month=month,
            floor=floor,
            **kwargs
        )
        test_dataset = LongTermWiFiDataset(
            data_root=data_root,
            split='test',
            download=download,
            month=month,
            floor=floor,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return LongTermWiFiDataset(
            data_root=data_root,
            split=split,
            download=download,
            month=month,
            floor=floor,
            **kwargs
        )


LongTermWiFi.list_months = LongTermWiFiDataset.list_months
LongTermWiFi.list_floors = LongTermWiFiDataset.list_floors

