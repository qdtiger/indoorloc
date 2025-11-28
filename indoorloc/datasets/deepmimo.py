"""
DeepMIMO Dataset Implementation

Ray-tracing based simulated CSI dataset supporting Massive MIMO
and mmWave scenarios with configurable indoor/outdoor environments.

Reference:
    DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave
    and Massive MIMO Applications
    https://www.deepmimo.net

Supports:
    - Dynamic scenario discovery (scan local directories)
    - Automatic download via official deepmimo package or direct URLs
    - Multiple scenarios loading
    - Complex-valued CSI data
"""
from pathlib import Path
from typing import Optional, Any, List, Dict, Union
import numpy as np
import warnings

from .base import CSIDataset
from ..signals.csi import CSISignal, CSIMetadata
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import get_data_home


# Known DeepMIMO scenarios with metadata
DEEPMIMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    'O1_60': {
        'name': 'O1_60',
        'environment': 'outdoor',
        'frequency_ghz': 60,
        'description': 'Outdoor street canyon at 60 GHz',
    },
    'O1_28': {
        'name': 'O1_28',
        'environment': 'outdoor',
        'frequency_ghz': 28,
        'description': 'Outdoor street canyon at 28 GHz',
    },
    'I3_60': {
        'name': 'I3_60',
        'environment': 'indoor',
        'frequency_ghz': 60,
        'description': 'Indoor conference room + corridor at 60 GHz',
    },
    'I3_28': {
        'name': 'I3_28',
        'environment': 'indoor',
        'frequency_ghz': 28,
        'description': 'Indoor conference room + corridor at 28 GHz',
    },
    'I1_60': {
        'name': 'I1_60',
        'environment': 'indoor',
        'frequency_ghz': 60,
        'description': 'Indoor office at 60 GHz',
    },
    'I2_60': {
        'name': 'I2_60',
        'environment': 'indoor',
        'frequency_ghz': 60,
        'description': 'Indoor hall at 60 GHz',
    },
    'O2_60': {
        'name': 'O2_60',
        'environment': 'outdoor',
        'frequency_ghz': 60,
        'description': 'Outdoor urban at 60 GHz',
    },
    'Boston5G': {
        'name': 'Boston5G',
        'environment': 'outdoor',
        'frequency_ghz': 28,
        'description': 'Boston downtown 5G at 28 GHz',
    },
    'asu_campus_3p5': {
        'name': 'asu_campus_3p5',
        'environment': 'outdoor',
        'frequency_ghz': 3.5,
        'description': 'ASU campus at 3.5 GHz',
    },
}


@DATASETS.register_module()
class DeepMIMODataset(CSIDataset):
    """DeepMIMO Ray-Tracing Simulated CSI Dataset.

    Ray-tracing based CSI generation using Remcom Wireless InSite.
    Features:
    - Dynamic scenario discovery from local directories
    - Automatic download via official deepmimo package
    - Multiple preset scenarios (O1_60, I3_60, Boston5G, etc.)
    - Complex-valued CSI data with CSISignal
    - 3D coordinates (x, y, z)

    Args:
        data_root: Root directory containing DeepMIMO scenarios.
        split: Dataset split ('train' or 'test').
        download: Whether to download the scenario if not found.
        scenario: Scenario name ('O1_60', 'I3_60', etc.), list of scenarios,
                  or 'all' to load all local scenarios.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.8).

    Example:
        >>> import indoorloc as iloc
        >>>
        >>> # List available scenarios
        >>> iloc.DeepMIMO.list_scenarios()
        ['O1_60', 'I3_60', 'Boston5G', ...]
        >>>
        >>> # Load single scenario
        >>> train, test = iloc.DeepMIMO(scenario='O1_60', download=True)
        >>>
        >>> # Load multiple scenarios
        >>> dataset = iloc.DeepMIMO(scenario=['O1_60', 'I3_60'])
        >>>
        >>> # Load all local scenarios
        >>> dataset = iloc.DeepMIMO(scenario='all')
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        scenario: Union[str, List[str]] = 'O1_60',
        transform: Optional[Any] = None,
        normalize: bool = False,
        normalize_method: str = 'standard',
        train_ratio: float = 0.8,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self._scenario_param = scenario
        self._num_antennas = 0
        self._num_subcarriers = 0

        # Determine scenarios to load
        if scenario == 'all':
            self._scenarios = self._discover_local_scenarios(data_root)
        elif isinstance(scenario, list):
            self._scenarios = scenario
        else:
            self._scenarios = [scenario]

        if not self._scenarios:
            raise ValueError("No scenarios specified or found")

        # Store primary scenario for metadata
        self.scenario = self._scenarios[0]

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
        return 'DeepMIMO'

    @property
    def num_antennas(self) -> int:
        return self._num_antennas

    @property
    def num_subcarriers(self) -> int:
        return self._num_subcarriers

    @classmethod
    def list_scenarios(cls, data_root: Optional[str] = None) -> List[str]:
        """List all available scenarios.

        Combines known scenarios from registry with locally discovered ones.

        Args:
            data_root: Root directory to scan for local scenarios.

        Returns:
            List of scenario names.
        """
        known = list(DEEPMIMO_SCENARIOS.keys())
        local = cls._discover_local_scenarios(data_root)

        # Combine and deduplicate
        all_scenarios = list(set(known + local))
        return sorted(all_scenarios)

    @classmethod
    def _discover_local_scenarios(cls, data_root: Optional[str] = None) -> List[str]:
        """Scan local directory for available scenarios.

        Args:
            data_root: Root directory to scan.

        Returns:
            List of discovered scenario names.
        """
        if data_root is None:
            root = get_data_home() / 'deepmimo'
        else:
            root = Path(data_root)

        if not root.exists():
            return []

        scenarios = []
        for item in root.iterdir():
            if item.is_dir():
                # Check for MAT files
                mat_files = list(item.glob('*.mat'))
                if mat_files:
                    scenarios.append(item.name)
            elif item.suffix == '.mat':
                # MAT file directly in root
                scenarios.append(item.stem)

        return scenarios

    @classmethod
    def get_scenario_info(cls, scenario: str) -> Dict[str, Any]:
        """Get metadata for a scenario.

        Args:
            scenario: Scenario name.

        Returns:
            Dictionary with scenario metadata.
        """
        if scenario in DEEPMIMO_SCENARIOS:
            return DEEPMIMO_SCENARIOS[scenario].copy()
        return {
            'name': scenario,
            'environment': 'unknown',
            'frequency_ghz': None,
            'description': f'User scenario: {scenario}',
        }

    def _check_exists(self) -> bool:
        """Check if at least one scenario exists."""
        for scenario in self._scenarios:
            scenario_dir = self.data_root / scenario
            if scenario_dir.exists():
                mat_files = list(scenario_dir.glob('*.mat'))
                if mat_files:
                    return True
            # Check for MAT file directly
            if (self.data_root / f'{scenario}.mat').exists():
                return True
        return False

    def _download(self) -> None:
        """Download scenarios using official deepmimo package."""
        self.data_root.mkdir(parents=True, exist_ok=True)

        for scenario in self._scenarios:
            scenario_dir = self.data_root / scenario
            if scenario_dir.exists() and list(scenario_dir.glob('*.mat')):
                print(f"场景 '{scenario}' 已存在")
                continue

            print(f"准备下载场景 '{scenario}'...")

            try:
                self._download_via_deepmimo(scenario)
            except ImportError:
                # 重新抛出 ImportError，让用户看到清晰的安装指引
                raise
            except Exception as e:
                warnings.warn(
                    f"下载 '{scenario}' 失败: {e}\n"
                    f"请手动下载: https://www.deepmimo.net/scenarios/"
                )

    def _download_via_deepmimo(self, scenario: str) -> None:
        """Download using official deepmimo package.

        Args:
            scenario: Scenario name to download.
        """
        try:
            import deepmimo as dm
        except ImportError:
            raise ImportError(
                "DeepMIMO 下载功能需要安装官方 deepmimo 包。\n\n"
                "安装方式（二选一）:\n"
                "  pip install indoorloc[deepmimo]   # 推荐\n"
                "  pip install DeepMIMO              # 或单独安装\n\n"
                "手动下载:\n"
                f"  访问 https://www.deepmimo.net/scenarios/ 下载 '{scenario}'\n"
                f"  解压到 {self.data_root / scenario}/"
            )

        # Download using official API
        print(f"正在下载场景 '{scenario}'...")
        print(f"使用 deepmimo.download('{scenario}')")
        dm.download(scenario)
        print(f"场景 '{scenario}' 下载完成")

    def _load_data(self) -> None:
        """Load data from all specified scenarios."""
        all_signals = []
        all_locations = []

        for scenario in self._scenarios:
            signals, locations = self._load_scenario(scenario)
            all_signals.extend(signals)
            all_locations.extend(locations)

        if not all_signals:
            warnings.warn("No data loaded, generating demo data")
            self._generate_demo_data()
            return

        # Apply train/test split
        n_samples = len(all_signals)
        n_train = int(n_samples * self.train_ratio)

        np.random.seed(42)
        indices = np.random.permutation(n_samples)

        if self.split == 'train':
            selected = indices[:n_train]
        else:
            selected = indices[n_train:]

        self._signals = [all_signals[i] for i in selected]
        self._locations = [all_locations[i] for i in selected]

        print(f"Loaded {len(self._signals)} samples from {len(self._scenarios)} scenario(s)")

    def _load_scenario(self, scenario: str) -> tuple:
        """Load a single scenario.

        Args:
            scenario: Scenario name.

        Returns:
            Tuple of (signals, locations) lists.
        """
        signals = []
        locations = []

        # Find MAT file(s)
        scenario_dir = self.data_root / scenario
        mat_file = self.data_root / f'{scenario}.mat'

        if scenario_dir.exists():
            mat_files = list(scenario_dir.glob('*.mat'))
        elif mat_file.exists():
            mat_files = [mat_file]
        else:
            warnings.warn(f"Scenario '{scenario}' not found")
            return signals, locations

        for mat_path in mat_files:
            s, l = self._load_mat_file(mat_path, scenario)
            signals.extend(s)
            locations.extend(l)

        return signals, locations

    def _load_mat_file(self, filepath: Path, scenario: str) -> tuple:
        """Load and parse a DeepMIMO MAT file.

        Args:
            filepath: Path to MAT file.
            scenario: Scenario name for metadata.

        Returns:
            Tuple of (signals, locations) lists.
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required. Install with: pip install scipy")

        signals = []
        locations = []

        print(f"Loading {filepath}...")
        data = loadmat(filepath, squeeze_me=True, struct_as_record=False)

        # DeepMIMO format can vary, try common structures
        channels = None
        user_locs = None

        # Try different key patterns
        for key in ['channel', 'channels', 'H', 'CSI', 'csi']:
            if key in data:
                channels = data[key]
                break

        for key in ['user_loc', 'user_locs', 'locations', 'loc', 'pos', 'position']:
            if key in data:
                user_locs = data[key]
                break

        # Handle DeepMIMO v2/v3 structure with 'user' field
        if 'user' in data:
            user_data = data['user']
            if hasattr(user_data, '__iter__'):
                channels_list = []
                locs_list = []
                for user in user_data:
                    if hasattr(user, 'channel'):
                        channels_list.append(user.channel)
                    if hasattr(user, 'loc'):
                        locs_list.append(user.loc)
                if channels_list:
                    channels = np.array(channels_list)
                if locs_list:
                    user_locs = np.array(locs_list)

        if channels is None:
            print(f"  Could not find channel data. Available keys: {list(data.keys())}")
            return signals, locations

        # Ensure channels is at least 2D
        if channels.ndim == 1:
            channels = channels.reshape(1, -1)

        n_samples = channels.shape[0]
        print(f"  Found {n_samples} samples")

        # Update antenna/subcarrier counts
        if channels.ndim >= 2:
            self._num_subcarriers = channels.shape[-1]
        if channels.ndim >= 3:
            self._num_antennas = channels.shape[1]
        else:
            self._num_antennas = 1

        # Get scenario metadata
        scenario_info = self.get_scenario_info(scenario)

        # Create samples
        for i in range(n_samples):
            csi = channels[i]

            # Create CSI metadata
            metadata = CSIMetadata(
                num_antennas=self._num_antennas,
                num_subcarriers=self._num_subcarriers,
                frequency_ghz=scenario_info.get('frequency_ghz'),
                scenario=scenario,
            )

            signal = CSISignal(csi_values=csi, metadata=metadata)

            # Create location
            if user_locs is not None and i < len(user_locs):
                loc = user_locs[i]
                if len(loc) >= 3:
                    x, y, z = loc[0], loc[1], loc[2]
                elif len(loc) >= 2:
                    x, y, z = loc[0], loc[1], 0.0
                else:
                    x, y, z = 0.0, 0.0, 0.0
            else:
                x, y, z = 0.0, 0.0, 0.0

            location = Location(
                coordinate=Coordinate(x=float(x), y=float(y), z=float(z)),
                floor=0,
                building_id=scenario
            )

            signals.append(signal)
            locations.append(location)

        return signals, locations

    def _generate_demo_data(self) -> None:
        """Generate demo data for testing when real data is not available."""
        np.random.seed(42 if self.split == 'train' else 123)

        # Demo parameters
        n_samples = 500
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        self._num_antennas = 64
        self._num_subcarriers = 64

        for _ in range(n):
            # Random indoor position
            x = np.random.uniform(0, 20)
            y = np.random.uniform(0, 15)
            z = np.random.uniform(0, 3)

            # Generate complex CSI (simplified channel model)
            csi_real = np.random.randn(self._num_antennas, self._num_subcarriers)
            csi_imag = np.random.randn(self._num_antennas, self._num_subcarriers)
            csi = (csi_real + 1j * csi_imag).astype(np.complex64)

            metadata = CSIMetadata(
                num_antennas=self._num_antennas,
                num_subcarriers=self._num_subcarriers,
                scenario='demo',
            )

            signal = CSISignal(csi_values=csi, metadata=metadata)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id='demo'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")
        print("Note: For real data, use download=True or visit https://www.deepmimo.net")

    def _normalize_signals(self) -> None:
        """Normalize CSI signals."""
        self._signals = [
            signal.normalize(method=self.normalize_method)
            for signal in self._signals
        ]


def DeepMIMO(
    data_root: Optional[str] = None,
    split: Optional[str] = None,
    download: bool = False,
    scenario: Union[str, List[str]] = 'O1_60',
    **kwargs
):
    """Convenience function for loading DeepMIMO dataset.

    Args:
        data_root: Root directory for data.
        split: 'train', 'test', 'all', or None (returns train, test tuple).
        download: Whether to download if not found.
        scenario: Scenario name(s) to load.
        **kwargs: Additional arguments for DeepMIMODataset.

    Returns:
        Dataset or tuple of (train, test) datasets.

    Example:
        >>> # Get train/test split
        >>> train, test = iloc.DeepMIMO(scenario='O1_60', download=True)
        >>>
        >>> # Get only train
        >>> train = iloc.DeepMIMO(scenario='O1_60', split='train')
        >>>
        >>> # List available scenarios
        >>> iloc.DeepMIMO.list_scenarios()
    """
    if split is None:
        train = DeepMIMODataset(
            data_root=data_root,
            split='train',
            download=download,
            scenario=scenario,
            **kwargs
        )
        test = DeepMIMODataset(
            data_root=data_root,
            split='test',
            download=download,
            scenario=scenario,
            **kwargs
        )
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = DeepMIMODataset(
            data_root=data_root,
            split='train',
            download=download,
            scenario=scenario,
            **kwargs
        )
        test = DeepMIMODataset(
            data_root=data_root,
            split='test',
            download=download,
            scenario=scenario,
            **kwargs
        )
        return ConcatDataset([train, test])
    else:
        return DeepMIMODataset(
            data_root=data_root,
            split=split,
            download=download,
            scenario=scenario,
            **kwargs
        )


# Attach class method to convenience function for list_scenarios access
DeepMIMO.list_scenarios = DeepMIMODataset.list_scenarios
