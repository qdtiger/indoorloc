"""
WiFi+Magnetic Hybrid Indoor Localization Dataset Implementation

Combines WiFi RSSI fingerprinting with magnetic field measurements
for robust indoor positioning in magnetically-rich environments.

Reference:
    Indoor Localization Using WiFi and Magnetic Field Fingerprinting.
    UCI Machine Learning Repository.

Dataset URL: https://archive.ics.uci.edu/dataset/626/wifi+magnetic+indoor+localization
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import HybridDataset
from ..signals.wifi import WiFiSignal
from ..signals.magnetometer import MagnetometerSignal
from ..signals.hybrid import HybridSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_uci


@DATASETS.register_module()
class WiFiMagneticHybridDataset(HybridDataset):
    """WiFi+Magnetic Hybrid Indoor Localization Dataset.

    Multi-modal dataset combining WiFi RSSI with magnetic field measurements
    for enhanced positioning accuracy, especially in environments with
    metallic structures and infrastructure.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/wifi_magnetic_hybrid).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        use_wifi: Whether to include WiFi signals (default: True).
        use_magnetic: Whether to include magnetic signals (default: True).
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> # Download WiFi+Magnetic hybrid dataset
        >>> dataset = iloc.WiFiMagneticHybrid(download=True, split='train')
        >>> # Access hybrid signal
        >>> signal = dataset[0]
        >>> wifi_signal = signal.wifi_signal
        >>> mag_signal = signal.magnetic_signal

    Dataset structure:
        data_root/
        └── wifi_magnetic_data.csv

    CSV format:
        - Columns: x, y, floor, wap1_rssi, ..., wapN_rssi, mag_x, mag_y, mag_z, mag_total
        - RSSI values in dBm
        - Magnetic field in μT (microtesla)
    """

    # UCI dataset name
    UCI_DATASET_NAME = 'wifi-magnetic-indoor-localization'

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # Required files
    REQUIRED_FILES = ['wifi_magnetic_data.csv']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        use_wifi: bool = True,
        use_magnetic: bool = True,
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.use_wifi = use_wifi
        self.use_magnetic = use_magnetic
        self.train_ratio = train_ratio
        self._num_waps = None

        if not use_wifi and not use_magnetic:
            raise ValueError("At least one of use_wifi or use_magnetic must be True")

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
        return 'WiFiMagneticHybrid'

    @property
    def num_waps(self) -> int:
        if self._num_waps is None:
            return 0
        return self._num_waps

    @property
    def signal_types(self) -> List[str]:
        """Return list of signal types in this dataset."""
        types = []
        if self.use_wifi:
            types.append('wifi')
        if self.use_magnetic:
            types.append('magnetic')
        return types

    def get_signal_modalities(self) -> List[str]:
        """Get list of signal modalities present in this dataset."""
        return self.signal_types

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download WiFi+Magnetic hybrid dataset from UCI."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading WiFi+Magnetic hybrid dataset from UCI...")

        try:
            download_from_uci(
                dataset_name=self.UCI_DATASET_NAME,
                root=self.data_root,
                filenames=self.REQUIRED_FILES,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download WiFi+Magnetic hybrid dataset: {e}\n"
                f"Please download manually from: "
                f"https://archive.ics.uci.edu/dataset/626/"
            )

    def _load_data(self) -> None:
        """Load WiFi+Magnetic hybrid dataset from CSV file."""
        filepath = self.data_root / 'wifi_magnetic_data.csv'

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load WiFi+Magnetic hybrid dataset.\n"
                "Install with: pip install pandas"
            )

        # Load CSV
        df = pd.read_csv(filepath)

        # Expected columns: x, y, floor, wap1_rssi, ..., wapN_rssi, mag_x, mag_y, mag_z, mag_total
        coord_cols = ['x', 'y', 'floor']
        mag_cols = ['mag_x', 'mag_y', 'mag_z', 'mag_total']

        # Identify WiFi columns (those not in coord_cols or mag_cols)
        wap_cols = [col for col in df.columns
                    if col not in coord_cols + mag_cols]

        self._num_waps = len(wap_cols)

        # Split data
        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:  # test
            df_split = df.iloc[num_train:]

        # Process each sample
        for idx, row in df_split.iterrows():
            # Extract location
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            floor = int(row.get('floor', 0))

            # Create WiFi signal
            wifi_signal = None
            if self.use_wifi and len(wap_cols) > 0:
                rssi_values = []
                for col in wap_cols:
                    rssi = row[col]
                    if pd.isna(rssi):
                        rssi = self.NOT_DETECTED_VALUE
                    rssi_values.append(float(rssi))
                wifi_signal = WiFiSignal(rssi_values=np.array(rssi_values))

            # Create magnetic signal
            mag_signal = None
            if self.use_magnetic:
                mag_x = float(row.get('mag_x', 0.0))
                mag_y = float(row.get('mag_y', 0.0))
                mag_z = float(row.get('mag_z', 0.0))

                # Create magnetic field measurement
                magnetic_field = np.array([mag_x, mag_y, mag_z])
                mag_signal = MagnetometerSignal(magnetic_field=magnetic_field)

            # Create hybrid signal
            if wifi_signal is not None or mag_signal is not None:
                hybrid_signal = HybridSignal(
                    wifi_signal=wifi_signal,
                    magnetic_signal=mag_signal
                )

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y),
                    floor=floor,
                    building_id='0'
                )

                self._signals.append(hybrid_signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from WiFi+Magnetic hybrid dataset ({self.split} split)")
        if self.use_wifi:
            print(f"WiFi APs: {self._num_waps}")
        if self.use_magnetic:
            print(f"Magnetic sensors: 3-axis magnetometer")


# Alias for convenience
WiFiMagneticHybrid = WiFiMagneticHybridDataset
