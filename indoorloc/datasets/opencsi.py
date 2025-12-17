"""
OpenCSI Dataset Implementation

LTE CSI fingerprint dataset collected using SDR terminal in indoor environment.
Robot-collected with high-density grid sampling along 8 trajectories.

Reference:
    Khatib, O., Jaouadi, R., and Moreira, J.
    OpenCSI: An Open-Source Dataset for Indoor Localization Using CSI-Based Fingerprinting
    arXiv:2104.07963, 2021

Dataset URL: https://figshare.com/articles/dataset/openCSI/19596379
DOI: 10.6084/m9.figshare.19596379.v1

Data Format:
    - LTE CSI spanning 1200 subcarriers
    - 4 antenna ports, 20 MHz bandwidth
    - 8 trajectories (5m each, 1cm spacing = 500 points per line)
    - Total: ~4000 reference points
"""
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Any, List
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


@DATASETS.register_module()
class OpenCSIDataset(WiFiDataset):
    """OpenCSI LTE CSI Dataset.

    3GPP LTE downlink CSI dataset collected using SDR terminal.
    Features:
    - Indoor single-cell environment (Swisscom Digital Lab, EPFL)
    - Wheeled robot automatic collection
    - 8 trajectories (5m long, 50cm apart, 1cm resolution)
    - 1200 LTE subcarriers across 4 antenna ports
    - 2D ground truth coordinates (x, y)

    Note:
        This is a large dataset (~2GB). Download may take significant time.

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        trajectory: Trajectory ID (1-8 or 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.OpenCSI(download=True)
        >>> signal, location = train[0]
    """

    DOWNLOAD_URL = 'https://ndownloader.figshare.com/files/34809589'
    EXPECTED_SIZE = 2030000000  # ~2GB

    NOT_DETECTED_VALUE = -100.0
    NUM_FEATURES = 1200  # LTE CSI features (1200 subcarriers)

    _download_message_shown = False  # Class variable to avoid duplicate messages

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        trajectory: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.trajectory = trajectory
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
        return 'OpenCSI'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        """Always returns True since demo data is available."""
        # Ensure data_root exists for demo data generation
        self.data_root.mkdir(parents=True, exist_ok=True)
        return True

    def _has_real_data(self) -> bool:
        """Check if real .mat files exist."""
        mat_files = list(self.data_root.glob('*.mat'))
        if mat_files:
            return True
        opencsi_dir = self.data_root / 'openCSI'
        if opencsi_dir.exists() and any(opencsi_dir.glob('*.mat')):
            return True
        return False

    def _download(self) -> None:
        """Download and extract OpenCSI dataset from Figshare."""
        if self._has_real_data():
            return

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_file = self.data_root / 'openCSI.zip'

        # Check if zip exists and is complete
        if zip_file.exists():
            current_size = zip_file.stat().st_size
            if current_size >= self.EXPECTED_SIZE * 0.9:  # Allow 10% tolerance
                print(f"Extracting OpenCSI ({current_size / 1e9:.2f} GB)...")
                self._extract_zip(zip_file)
                return

        # Show download instructions only once
        if not OpenCSIDataset._download_message_shown:
            OpenCSIDataset._download_message_shown = True

            incomplete_msg = ""
            if zip_file.exists():
                current_size = zip_file.stat().st_size
                incomplete_msg = f"Incomplete download detected: {current_size / 1e6:.1f} MB / ~2 GB\n\n"

            print("\n" + "=" * 70)
            print("OpenCSI: Large Dataset (~2 GB) - Manual Download Required")
            print("=" * 70)
            print(f"""
{incomplete_msg}This is an LTE CSI fingerprint dataset with 1200 subcarriers.

Download options:

  1. Web browser:
     https://doi.org/10.6084/m9.figshare.19596379.v1

  2. Command line:
     curl -L -o '{zip_file}' \\
       '{self.DOWNLOAD_URL}'

  3. Then extract:
     cd {self.data_root} && unzip openCSI.zip

Using demo data for now...
""")
            print("=" * 70 + "\n")

    def _extract_zip(self, zip_file: Path) -> None:
        """Extract the downloaded zip file."""
        print(f"Extracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(self.data_root)
            print("Extraction complete")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file.name} is corrupted. Please re-download.")
            raise RuntimeError(f"Corrupted zip file: {zip_file}")

    def _load_data(self) -> None:
        """Load OpenCSI dataset from available files."""
        # Search for .mat files in multiple locations
        mat_files = list(self.data_root.glob('*.mat'))

        # Also check for extracted subdirectory
        opencsi_dir = self.data_root / 'openCSI'
        if opencsi_dir.exists():
            mat_files.extend(opencsi_dir.glob('*.mat'))

        if mat_files:
            self._load_from_mat_files(mat_files)
        else:
            self._generate_demo_data()

    def _load_from_mat_files(self, mat_files: List[Path]) -> None:
        """Load data from MATLAB .mat files."""
        try:
            from scipy.io import loadmat
        except ImportError:
            try:
                import h5py
            except ImportError:
                raise ImportError("scipy or h5py required. Install with: pip install scipy h5py")

        all_samples = []

        for mat_file in mat_files:
            try:
                data = self._load_single_mat(mat_file)
                if data is not None:
                    all_samples.extend(data)
            except Exception as e:
                print(f"Warning: Failed to load {mat_file.name}: {e}")
                continue

        if not all_samples:
            print("No valid data loaded from .mat files")
            self._generate_demo_data()
            return

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(all_samples)

        num_train = int(len(all_samples) * self.train_ratio)
        if self.split == 'train':
            samples = all_samples[:num_train]
        else:
            samples = all_samples[num_train:]

        for sample in samples:
            signal = WiFiSignal(rssi_values=sample['csi'])
            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=0,
                building_id=str(sample.get('trajectory', '0'))
            )
            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from OpenCSI (split={self.split})")

    def _load_single_mat(self, filepath: Path) -> List[dict]:
        """Load a single .mat file and return list of samples."""
        samples = []

        # Try scipy.io first
        try:
            from scipy.io import loadmat
            data = loadmat(str(filepath))
        except Exception:
            # Try h5py for HDF5 format
            import h5py
            with h5py.File(str(filepath), 'r') as f:
                data = {k: np.array(v) for k, v in f.items()}

        # Extract CSI and position data based on common key patterns
        csi_data = None
        x_data = None
        y_data = None

        for key in data.keys():
            key_lower = key.lower()
            if 'csi' in key_lower or 'feature' in key_lower:
                csi_data = np.array(data[key])
            elif key_lower == 'x' or 'coord_x' in key_lower or 'pos_x' in key_lower:
                x_data = np.array(data[key]).flatten()
            elif key_lower == 'y' or 'coord_y' in key_lower or 'pos_y' in key_lower:
                y_data = np.array(data[key]).flatten()

        if csi_data is None or x_data is None or y_data is None:
            return samples

        # Handle different array shapes
        if csi_data.ndim == 2:
            if csi_data.shape[1] == self.NUM_FEATURES:
                pass  # Already (N, features)
            elif csi_data.shape[0] == self.NUM_FEATURES:
                csi_data = csi_data.T  # Transpose to (N, features)

        n_samples = min(len(csi_data), len(x_data), len(y_data))

        for i in range(n_samples):
            samples.append({
                'csi': csi_data[i].astype(np.float32),
                'x': float(x_data[i]),
                'y': float(y_data[i]),
            })

        return samples

    def _load_from_csv(self, filepath: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        df = pd.read_csv(filepath)
        num_train = int(len(df) * self.train_ratio)

        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:
            df_split = df.iloc[num_train:]

        for idx, row in df_split.iterrows():
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))

            feature_cols = [c for c in df_split.columns if c not in ['x', 'y', 'trajectory']]
            rssi_values = np.array([float(row[c]) for c in feature_cols[:self.NUM_FEATURES]])

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples ({self.split} split)")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        n_samples = 800
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for i in range(n):
            trajectory_id = (i % 8) + 1
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)

            rssi_values = np.random.uniform(-100, -40, self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id=str(trajectory_id)
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")


def OpenCSI(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading OpenCSI dataset."""
    if split is None:
        train = OpenCSIDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = OpenCSIDataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = OpenCSIDataset(data_root=data_root, split='train', download=download, **kwargs)
        test = OpenCSIDataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return OpenCSIDataset(data_root=data_root, split=split, download=download, **kwargs)
