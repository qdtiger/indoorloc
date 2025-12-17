"""
H-WILD (Human-held WiFi Indoor Localization Dataset) Implementation

WiFi CSI dataset collected with human-held devices, providing realistic
indoor localization data with UWB ground truth coordinates.

Reference:
    H-WILD: Human-held device WiFi indoor localization dataset
    https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset

Data Format:
    .mat files containing:
    - features_csi: CSI features (N × 90), 3 antennas × 30 subcarriers
    - features_rssi: RSSI values (N × 3)
    - uwb_coordinate_x, uwb_coordinate_y: UWB ground truth coordinates
"""
import json
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


GITHUB_API_BASE = 'https://api.github.com/repos/H-WILD/human_held_device_wifi_indoor_localization_dataset/contents'
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/H-WILD/human_held_device_wifi_indoor_localization_dataset/main'

# Environment folder mapping (code name -> GitHub folder name)
ENVIRONMENT_FOLDERS = {
    'conference': 'Conference',
    'laboratory': 'Laboratory',
    'office': 'Office',
    'lounge': 'Lounge',
}


@DATASETS.register_module()
class HWILDDataset(WiFiDataset):
    """H-WILD (Human-held WiFi Indoor Localization Dataset).

    WiFi CSI dataset collected using commercial WiFi APs with human-held
    terminal devices. Features:
    - CSI dimension: 90 (3 antennas × 30 subcarriers)
    - 4 indoor environments: Conference, Laboratory, Office, Lounge
    - 10 volunteers with diverse holding positions
    - ~120,000 samples total
    - UWB-provided ground truth coordinates (centimeter-level accuracy)

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        environment: Indoor environment ('conference', 'laboratory', 'office', 'lounge', 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.HWILD(download=True)
        >>> signal, location = train[0]
    """

    NOT_DETECTED_VALUE = -80.0
    NUM_FEATURES = 90  # 3 antennas × 30 subcarriers

    ENVIRONMENTS = ['conference', 'laboratory', 'office', 'lounge']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        environment: str = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        seed: int = 42,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.seed = seed

        # Parse environment parameter
        if environment == 'all':
            self._environments = self.ENVIRONMENTS.copy()
        elif isinstance(environment, list):
            self._environments = [e.lower() for e in environment]
        else:
            self._environments = [environment.lower()]

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
        return 'hwild'

    @property
    def num_aps(self) -> int:
        return self.NUM_FEATURES

    def _check_exists(self) -> bool:
        """Check if dataset .mat files exist."""
        for env in self._environments:
            env_dir = self.data_root / env
            if not env_dir.exists():
                return False
            mat_files = list(env_dir.glob('*.mat'))
            if len(mat_files) == 0:
                return False
        return True

    def _download(self) -> None:
        """Download H-WILD dataset from GitHub."""
        if self._check_exists():
            print(f"H-WILD dataset already exists at {self.data_root}")
            return

        print("Downloading H-WILD dataset from GitHub...")
        self.data_root.mkdir(parents=True, exist_ok=True)

        total_downloaded = 0
        total_failed = 0

        for env in self._environments:
            github_folder = ENVIRONMENT_FOLDERS.get(env, env.capitalize())
            env_dir = self.data_root / env
            env_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Downloading {env} environment...")

            # Get file list from GitHub API
            api_url = f"{GITHUB_API_BASE}/{github_folder}"
            try:
                result = subprocess.run(
                    ['curl', '-sL', api_url],
                    capture_output=True, text=True, timeout=30
                )
                files = json.loads(result.stdout)

                if not isinstance(files, list):
                    print(f"    Warning: Could not list files for {env}")
                    continue

                mat_files = [f['name'] for f in files if f['name'].endswith('.mat')]
                print(f"    Found {len(mat_files)} .mat files")

                for i, filename in enumerate(mat_files):
                    dst = env_dir / filename
                    if dst.exists() and dst.stat().st_size > 1000:
                        total_downloaded += 1
                        continue

                    url = f"{GITHUB_RAW_BASE}/{github_folder}/{filename}"
                    subprocess.run(
                        ['curl', '-sL', '-o', str(dst), url],
                        capture_output=True, timeout=60
                    )

                    if dst.exists() and dst.stat().st_size > 1000:
                        total_downloaded += 1
                    else:
                        total_failed += 1

                    if (i + 1) % 10 == 0:
                        print(f"      Progress: {i+1}/{len(mat_files)}")

            except Exception as e:
                print(f"    Error downloading {env}: {e}")
                total_failed += 1

        print(f"Download complete: {total_downloaded} files downloaded, {total_failed} failed")

        if not self._check_exists():
            raise RuntimeError(
                f"Failed to download H-WILD dataset.\n"
                f"Please download manually from:\n"
                f"  https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset\n"
                f"Place .mat files in: {self.data_root}/<environment>/"
            )

    def _load_mat_file(self, mat_file: Path) -> Dict:
        """Load a single .mat file, handling both v7 and v7.3 formats."""
        # Try h5py first (for MATLAB v7.3 HDF5 format)
        try:
            import h5py
            with h5py.File(str(mat_file), 'r') as f:
                data = {}
                for key in f.keys():
                    data[key] = np.array(f[key])
                return data
        except Exception:
            pass

        # Fall back to scipy.io.loadmat (for older MATLAB formats)
        try:
            from scipy.io import loadmat
            return loadmat(str(mat_file))
        except Exception as e:
            raise RuntimeError(f"Cannot read {mat_file.name}: {e}")

    def _load_data(self) -> None:
        """Load H-WILD dataset from .mat files."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for H-WILD dataset. Install with: pip install h5py")

        all_samples: List[Dict] = []

        for env in self._environments:
            env_dir = self.data_root / env
            if not env_dir.exists():
                continue

            mat_files = sorted(env_dir.glob('*.mat'))

            for mat_file in mat_files:
                try:
                    data = self._load_mat_file(mat_file)

                    # Extract data arrays
                    features_csi = data.get('features_csi', None)
                    uwb_x = data.get('uwb_coordinate_x', None)
                    uwb_y = data.get('uwb_coordinate_y', None)

                    if features_csi is None or uwb_x is None or uwb_y is None:
                        continue

                    # Handle complex CSI data (structured array with 'real' and 'imag' fields)
                    features_csi = np.array(features_csi)
                    if features_csi.dtype.names and 'real' in features_csi.dtype.names:
                        # Extract magnitude from complex numbers
                        magnitude = np.sqrt(features_csi['real']**2 + features_csi['imag']**2)
                        features_csi = magnitude

                    # Handle HDF5 format (data is transposed: 90 x N -> N x 90)
                    if features_csi.ndim == 2 and features_csi.shape[0] == 90:
                        features_csi = features_csi.T
                    features_csi = features_csi.reshape(-1, 90) if features_csi.size > 0 else np.array([])

                    uwb_x = np.array(uwb_x).flatten()
                    uwb_y = np.array(uwb_y).flatten()

                    # Ensure all arrays have same length
                    n_samples = min(len(features_csi), len(uwb_x), len(uwb_y))

                    for i in range(n_samples):
                        all_samples.append({
                            'csi': features_csi[i].astype(np.float32),
                            'x': float(uwb_x[i]),
                            'y': float(uwb_y[i]),
                            'environment': env,
                            'file': mat_file.name,
                        })

                except Exception as e:
                    print(f"Warning: Failed to load {mat_file.name}: {e}")
                    continue

        if len(all_samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.data_root}")

        # Shuffle with fixed seed for reproducibility
        np.random.seed(self.seed)
        np.random.shuffle(all_samples)

        # Split train/test
        n_train = int(len(all_samples) * self.train_ratio)
        if self.split == 'train':
            samples = all_samples[:n_train]
        else:
            samples = all_samples[n_train:]

        # Create signals and locations
        for sample in samples:
            # Handle inf/nan values
            csi_values = np.nan_to_num(sample['csi'], nan=0.0, posinf=0.0, neginf=0.0)

            signal = WiFiSignal(rssi_values=csi_values)
            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=0,
                building_id=sample['environment']
            )

            self._signals.append(signal)
            self._locations.append(location)
            self._metadata.append({
                'environment': sample['environment'],
                'file': sample['file']
            })

        envs_str = ','.join(self._environments)
        print(f"Loaded {len(self._signals)} samples from H-WILD (environments={envs_str}, split={self.split})")


def HWILD(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading H-WILD dataset.

    Returns:
        - If split is None: Returns tuple (train_dataset, test_dataset)
        - If split is 'all': Returns merged train+test dataset
        - Otherwise: Returns single dataset for specified split
    """
    if split is None:
        train_dataset = HWILDDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = HWILDDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = HWILDDataset(
            data_root=data_root, split='train', download=download, **kwargs
        )
        test_dataset = HWILDDataset(
            data_root=data_root, split='test', download=download, **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return HWILDDataset(
            data_root=data_root, split=split, download=download, **kwargs
        )
