"""
HALOC Dataset Implementation

WiFi CSI trajectory dataset collected in indoor corridor environment
using ESP32-S3 with directional antenna.

Reference:
    Strohmayer, J., and Kampel, M. (2024).
    WiFi CSI-based Long-Range Person Localization Using Directional Antennas
    The Second Tiny Papers Track at ICLR 2024

Dataset URL: https://zenodo.org/records/10715595
GitHub: https://github.com/StrohmayerJ/HALOC

Data Format:
    CSV files containing:
    - data: CSI values as comma-separated complex numbers (real,imag pairs)
    - x, y, z: 3D coordinates
    6 trajectory files (0.csv to 5.csv)
    - Training: 0.csv, 1.csv, 2.csv, 3.csv
    - Validation: 4.csv
    - Test: 5.csv
"""
import subprocess
from pathlib import Path
from typing import Optional, Any, List
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


# Valid L-LTF subcarrier indices (52 subcarriers, excluding pilots and guard bands)
VALID_SUBCARRIERS = [
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 58, 59, 60, 61
]


@DATASETS.register_module()
class HALOCDataset(WiFiDataset):
    """HALOC WiFi CSI Trajectory Dataset.

    WiFi CSI dataset collected using ESP32-S3 with directional antenna.
    Features:
    - Indoor long corridor environment (~30m)
    - Single person walking along 6 trajectories
    - Strong multipath environment
    - 3D coordinates (x, y, z) per CSI packet
    - 52 subcarriers (L-LTF)

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train', 'val', or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.HALOC(download=True)
        >>> signal, location = train[0]
    """

    ZENODO_URL = 'https://zenodo.org/records/10715595/files'

    NOT_DETECTED_VALUE = 0.0
    NUM_FEATURES = 52  # 52 valid L-LTF subcarriers

    # Dataset splits based on official split
    TRAIN_FILES = ['0.csv', '1.csv', '2.csv', '3.csv']
    VAL_FILE = '4.csv'
    TEST_FILE = '5.csv'

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
        return 'HALOC'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        """Check if all required CSV files exist."""
        all_files = self.TRAIN_FILES + [self.VAL_FILE, self.TEST_FILE]
        return all((self.data_root / f).exists() for f in all_files)

    def _download(self) -> None:
        """Download HALOC dataset from Zenodo."""
        if self._check_exists():
            print(f"HALOC dataset already exists at {self.data_root}")
            return

        print("Downloading HALOC dataset from Zenodo...")
        self.data_root.mkdir(parents=True, exist_ok=True)

        all_files = self.TRAIN_FILES + [self.VAL_FILE, self.TEST_FILE]
        downloaded = 0

        for filename in all_files:
            dst = self.data_root / filename
            if dst.exists() and dst.stat().st_size > 1000:
                downloaded += 1
                continue

            url = f"{self.ZENODO_URL}/{filename}?download=1"
            print(f"  Downloading {filename}...")

            try:
                subprocess.run(
                    ['curl', '-sL', '-o', str(dst), url],
                    capture_output=True, timeout=120
                )
                if dst.exists() and dst.stat().st_size > 1000:
                    downloaded += 1
                    print(f"    Downloaded: {dst.stat().st_size / 1024:.1f} KB")
            except Exception as e:
                print(f"    Error downloading {filename}: {e}")

        print(f"Download complete: {downloaded}/{len(all_files)} files")

        if not self._check_exists():
            raise RuntimeError(
                f"Failed to download HALOC dataset.\n"
                f"Please download manually from:\n"
                f"  https://zenodo.org/records/10715595\n"
                f"Place CSV files (0.csv to 5.csv) in: {self.data_root}"
            )

    def _parse_csi_string(self, csi_str: str) -> np.ndarray:
        """Parse CSI string to amplitude values.

        CSI data is stored as JSON array of integers representing
        alternating imaginary and real parts of complex numbers.
        Format: "[imag0,real0,imag1,real1,...]"
        """
        try:
            import json
            # Parse JSON array format
            if csi_str.startswith('['):
                values = json.loads(csi_str)
            else:
                values = [int(v) for v in csi_str.split(',')]
        except (ValueError, AttributeError, json.JSONDecodeError):
            return np.zeros(self.NUM_FEATURES, dtype=np.float32)

        # Extract complex values for valid subcarriers
        # Data format: pairs of (imag, real) for each subcarrier index
        amplitudes = []
        for idx in VALID_SUBCARRIERS:
            imag_idx = idx * 2
            real_idx = idx * 2 + 1
            if real_idx < len(values):
                imag = values[imag_idx]
                real = values[real_idx]
                amplitude = np.sqrt(float(real)**2 + float(imag)**2)
                amplitudes.append(amplitude)
            else:
                amplitudes.append(0.0)

        return np.array(amplitudes, dtype=np.float32)

    def _load_data(self) -> None:
        """Load HALOC dataset from CSV files."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        # Determine which files to load based on split
        if self.split == 'train':
            files_to_load = self.TRAIN_FILES
        elif self.split == 'val':
            files_to_load = [self.VAL_FILE]
        else:  # test
            files_to_load = [self.TEST_FILE]

        for filename in files_to_load:
            filepath = self.data_root / filename
            if not filepath.exists():
                print(f"Warning: {filename} not found, skipping")
                continue

            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                # Parse CSI data
                csi_data = row.get('data', '')
                amplitudes = self._parse_csi_string(str(csi_data))

                # Get coordinates
                x = float(row.get('x', 0.0))
                y = float(row.get('y', 0.0))
                z = float(row.get('z', 0.0))

                signal = WiFiSignal(rssi_values=amplitudes)
                location = Location(
                    coordinate=Coordinate(x=x, y=y, z=z),
                    floor=0,
                    building_id=filename.replace('.csv', '')
                )

                self._signals.append(signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from HALOC (split={self.split})")


def HALOC(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading HALOC dataset.

    Returns:
        - If split is None: Returns tuple (train_dataset, test_dataset)
        - If split is 'all': Returns merged train+val+test dataset
        - Otherwise: Returns single dataset for specified split ('train', 'val', 'test')
    """
    if split is None:
        train = HALOCDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = HALOCDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = HALOCDataset(data_root=data_root, split='train', download=download, **kwargs)
        val = HALOCDataset(data_root=data_root, split='val', download=download, **kwargs)
        test = HALOCDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, val, test])
    else:
        return HALOCDataset(data_root=data_root, split=split, download=download, **kwargs)
