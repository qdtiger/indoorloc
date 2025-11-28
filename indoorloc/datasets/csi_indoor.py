"""
CSI (Channel State Information) Indoor Localization Dataset Implementation

WiFi CSI-based indoor positioning dataset with fine-grained channel
information for precise location estimation.

Reference:
    CSI Indoor Localization Dataset.
    GitHub Repository.
    https://github.com/csi-positioning/indoor-dataset

Dataset URL: https://raw.githubusercontent.com/CSI-Positioning/IndoorDataset/main
"""
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
class CSIIndoorDataset(WiFiDataset):
    """CSI Indoor Localization Dataset.

    WiFi Channel State Information (CSI) based indoor positioning with
    amplitude and phase information from multiple subcarriers.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/csi_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> dataset = iloc.CSIIndoor(download=True, split='train')
        >>> signal, location = dataset[0]

    Dataset structure:
        data_root/
        └── csi_measurements.csv

    CSV format:
        - Columns: x, y, floor, csi_amp_1, ..., csi_amp_N, csi_phase_1, ..., csi_phase_N
        - CSI amplitude and phase for N subcarriers
    """

    # GitHub raw content base URL
    BASE_URL = 'https://raw.githubusercontent.com/CSI-Positioning/IndoorDataset/main'

    # Dataset constants
    NOT_DETECTED_VALUE = -80.0
    NUM_SUBCARRIERS = 52  # Typical for WiFi 802.11n

    # Required files
    REQUIRED_FILES = ['csi_measurements.csv']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        floor: Union[int, List[int], str] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
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
        return 'CSIIndoor'

    @classmethod
    def list_floors(cls, data_root: Optional[str] = None) -> List[int]:
        """List all available floors."""
        from ..utils.download import get_data_home
        if data_root is None:
            root = get_data_home() / 'csi_indoor'
        else:
            root = Path(data_root)
        data_file = root / 'csi_measurements.csv'
        if not data_file.exists():
            return []
        try:
            import pandas as pd
            df = pd.read_csv(data_file, usecols=['floor'])
            return sorted(df['floor'].astype(int).unique().tolist())
        except Exception:
            return []

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download CSI indoor dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading CSI indoor dataset from GitHub...")

        for filename in self.REQUIRED_FILES:
            url = f"{self.BASE_URL}/{filename}"
            try:
                download_url(
                    url=url,
                    root=self.data_root,
                    filename=filename,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {filename}: {e}\n"
                    f"Please download manually from: {self.BASE_URL}"
                )

    def _load_data(self) -> None:
        """Load CSI indoor dataset from CSV file."""
        filepath = self.data_root / 'csi_measurements.csv'

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load CSI indoor dataset.\n"
                "Install with: pip install pandas"
            )

        df = pd.read_csv(filepath)

        # Store available floors
        if 'floor' in df.columns:
            self._available_floors = sorted(df['floor'].astype(int).unique().tolist())

        # Filter by floor
        if self._floor_param != 'all' and 'floor' in df.columns:
            if isinstance(self._floor_param, int):
                selected = [self._floor_param]
            else:
                selected = list(self._floor_param)
            df = df[df['floor'].astype(int).isin(selected)]

        if len(df) == 0:
            raise ValueError(f"No data for floor={self._floor_param}")

        # Split data
        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        # Identify CSI amplitude columns (use as RSSI-like features)
        coord_cols = ['x', 'y', 'floor']
        csi_amp_cols = [col for col in df_split.columns if 'csi_amp' in col.lower()]
        self._num_waps = len(csi_amp_cols)

        # Process each sample
        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            floor = int(row.get('floor', 0))

            # Extract CSI amplitude values (use as signal strength features)
            rssi_values = []
            for col in csi_amp_cols:
                val = row[col]
                if pd.isna(val):
                    val = self.NOT_DETECTED_VALUE
                rssi_values.append(float(val))

            signal = WiFiSignal(rssi_values=np.array(rssi_values))
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=floor,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        floor_info = f" (floor: {self._floor_param})" if self._floor_param != 'all' else ""
        print(f"Loaded {len(self._signals)} samples from CSI indoor dataset{floor_info}")
        print(f"CSI subcarriers used: {self._num_waps}")



def CSIIndoor(data_root=None, split=None, download=False, floor='all', **kwargs):
    """
    Convenience function for loading CSIIndoor dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        floor: Floor(s) to load ('all', single int, or list)
        **kwargs: Additional arguments passed to CSIIndoorDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> train, test = CSIIndoor(download=True)
        >>> train = CSIIndoor(floor=[0, 1], split='train')
        >>> CSIIndoor.list_floors()
    """
    if split is None:
        train_dataset = CSIIndoorDataset(
            data_root=data_root, split='train', download=download,
            floor=floor, **kwargs
        )
        test_dataset = CSIIndoorDataset(
            data_root=data_root, split='test', download=download,
            floor=floor, **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = CSIIndoorDataset(
            data_root=data_root, split='train', download=download,
            floor=floor, **kwargs
        )
        test_dataset = CSIIndoorDataset(
            data_root=data_root, split='test', download=download,
            floor=floor, **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return CSIIndoorDataset(
            data_root=data_root, split=split, download=download,
            floor=floor, **kwargs
        )


# Attach class method to convenience function
CSIIndoor.list_floors = CSIIndoorDataset.list_floors

