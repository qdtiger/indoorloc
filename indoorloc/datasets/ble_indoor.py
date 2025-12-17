"""
BBIL (BLE Beacon Indoor Localization) Dataset Implementation

This module backs IndoorLoc's `BLEIndoorDataset` using the BBIL dataset
(co60ca/BBIL). The previously referenced BLE-Indoor-Positioning/Dataset
URLs are no longer available (404).

Reference:
    Kennedy, M., Spachos, P., Taylor, G.W.
    BLE Beacon Indoor Localization Dataset.
    https://github.com/co60ca/BBIL

Download (v0.9.0):
    https://github.com/co60ca/BBIL/releases/download/v0.9.0/experiment.zip

Experiments:
    - experiment1: office (11x25m)
    - experiment2: lab (11x7m)

Each split contains `data_wide.csv` with:
    - Datetime, beaconid
    - rssi_edge{N} columns (RSSI at each receiver/edge)
    - realx, realy: ground-truth coordinates
"""
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import numpy as np

from .base import BLEDataset
from ..signals.ble import BLESignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class BLEIndoorDataset(BLEDataset):
    """BBIL BLE Beacon Indoor Localization dataset.

    Loads the BBIL experiments (co60ca/BBIL) from the v0.9.0 release archive.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ble_indoor).
        split: Dataset split ('train', 'valid'/'val', or 'test').
        download: Whether to download the dataset if not found.
        floor: Backward-compatible alias for experiment selection:
            - 1 -> experiment1
            - 2 -> experiment2
            - 'all' -> both experiments
        experiment: Experiment selection: 'experiment1', 'experiment2', 'all', or list.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> dataset = iloc.BLEIndoor(download=True, experiment='experiment1', split='train')

    Dataset structure (after extraction):
        data_root/
        ├── experiment1/
        │   ├── train/{timestamp}_data_wide.csv (multiple files)
        │   ├── valid/{timestamp}_data_wide.csv
        │   └── test/{timestamp}_data_wide.csv
        └── experiment2/
            ├── train/{timestamp}_data_wide.csv (multiple files)
            ├── valid/{timestamp}_data_wide.csv
            └── test/{timestamp}_data_wide.csv

    CSV format:
        - Datetime, beaconid
        - rssi_edge{N}: RSSI in dBm (NaN/missing means not detected)
        - realx, realy: ground-truth coordinates in meters
    """

    DOWNLOAD_URL = 'https://github.com/co60ca/BBIL/releases/download/v0.9.0/experiment.zip'
    ZIP_FILENAME = 'experiment.zip'
    CSV_FILENAME = 'data_wide.csv'

    BASE_URL = DOWNLOAD_URL

    NOT_DETECTED_VALUE = -100.0

    AVAILABLE_EXPERIMENTS = ['experiment1', 'experiment2']
    FLOOR_TO_EXPERIMENT = {1: 'experiment1', 2: 'experiment2'}
    EXPERIMENT_TO_FLOOR = {'experiment1': 1, 'experiment2': 2}

    AVAILABLE_FLOORS = [1, 2]

    # BBIL uses "edge_N" column names (not "rssi_edge{N}")
    _EDGE_COL_RE = re.compile(r'^(?:rssi_)?edge[_]?(\d+)$', re.IGNORECASE)

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        floor: Union[int, List[int], str] = 'all',
        experiment: Union[str, List[str]] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        split = self._normalize_split(split)

        self._experiments = self._parse_experiments(experiment=experiment, floor=floor)
        self._floors = [self.EXPERIMENT_TO_FLOOR[e] for e in self._experiments]

        self.experiment = self._experiments[0] if len(self._experiments) == 1 else None
        self.floor = self._floors[0] if len(self._floors) == 1 else None

        self._num_beacons = None
        self._beacon_ids: Optional[List[str]] = None

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
        return 'BLEIndoor'

    @property
    def num_beacons(self) -> int:
        if self._num_beacons is None:
            return 0
        return self._num_beacons

    @classmethod
    def list_floors(cls) -> List[int]:
        """List all available legacy floor IDs (mapped to experiments)."""
        return cls.AVAILABLE_FLOORS.copy()

    @classmethod
    def list_experiments(cls) -> List[str]:
        """List all available experiments."""
        return cls.AVAILABLE_EXPERIMENTS.copy()

    @staticmethod
    def _normalize_split(split: str) -> str:
        s = (split or '').strip().lower()
        if s in ('train', 'test'):
            return s
        if s in ('val', 'valid', 'validation'):
            return 'valid'
        raise ValueError("split must be one of: 'train', 'valid'/'val', 'test'")

    @classmethod
    def _parse_experiments(
        cls,
        *,
        experiment: Union[str, List[str]],
        floor: Union[int, List[int], str],
    ) -> List[str]:
        use_experiment = experiment is not None and experiment != 'all'

        selected: List[str]
        if use_experiment:
            if isinstance(experiment, list):
                selected = [str(e) for e in experiment]
            else:
                selected = [str(experiment)]
        else:
            if floor is None or floor == 'all':
                selected = cls.AVAILABLE_EXPERIMENTS.copy()
            elif isinstance(floor, int):
                if floor not in cls.FLOOR_TO_EXPERIMENT:
                    raise ValueError(f"Unknown floor={floor}; valid: {sorted(cls.FLOOR_TO_EXPERIMENT.keys())}")
                selected = [cls.FLOOR_TO_EXPERIMENT[floor]]
            elif isinstance(floor, list):
                selected = []
                for f in floor:
                    if isinstance(f, str) and f.lower().startswith('experiment'):
                        selected.append(f)
                        continue
                    fi = int(f)
                    if fi not in cls.FLOOR_TO_EXPERIMENT:
                        raise ValueError(f"Unknown floor={fi}; valid: {sorted(cls.FLOOR_TO_EXPERIMENT.keys())}")
                    selected.append(cls.FLOOR_TO_EXPERIMENT[fi])
            elif isinstance(floor, str):
                if floor.lower() == 'all':
                    selected = cls.AVAILABLE_EXPERIMENTS.copy()
                else:
                    selected = [floor]
            else:
                selected = cls.AVAILABLE_EXPERIMENTS.copy()

        out: List[str] = []
        seen = set()
        for exp in selected:
            if str(exp).lower() == 'all':
                for e in cls.AVAILABLE_EXPERIMENTS:
                    if e not in seen:
                        out.append(e)
                        seen.add(e)
                continue

            match = next((e for e in cls.AVAILABLE_EXPERIMENTS if e.lower() == str(exp).lower()), None)
            if match is None:
                raise ValueError(f"Unknown experiment={exp}; valid: {cls.AVAILABLE_EXPERIMENTS + ['all']}")
            if match not in seen:
                out.append(match)
                seen.add(match)

        return out

    def _candidate_roots(self) -> List[Path]:
        """Possible roots where the zip may have extracted."""
        candidates = [self.data_root]
        for name in ('experiment', 'experiments', 'data'):
            p = self.data_root / name
            if p.exists():
                candidates.append(p)
        return candidates

    def _find_experiment_dir(self, experiment: str) -> Optional[Path]:
        for root in self._candidate_roots():
            p = root / experiment
            if p.is_dir():
                return p

        for p in self.data_root.glob(f'**/{experiment}'):
            if p.is_dir():
                return p

        return None

    @classmethod
    def _split_dir_candidates(cls, split: str) -> List[str]:
        s = cls._normalize_split(split)
        if s == 'valid':
            return ['valid', 'val', 'validation']
        return [s]

    def _find_data_wide_csvs(self, experiment: str, split: str) -> List[Path]:
        """Find all *_data_wide.csv files for a given experiment and split."""
        exp_dir = self._find_experiment_dir(experiment)
        if exp_dir is None:
            return []

        split_dirs = self._split_dir_candidates(split)
        for sd in split_dirs:
            split_path = exp_dir / sd
            if split_path.is_dir():
                # Find all *_data_wide.csv files in this split directory
                files = list(split_path.glob('*_data_wide.csv'))
                if files:
                    return sorted(files)

        # Fallback: search recursively
        all_files = list(exp_dir.rglob('*_data_wide.csv'))
        for sd in split_dirs:
            matching = [p for p in all_files if p.parent.name.lower() == sd.lower()]
            if matching:
                return sorted(matching)

        return []

    def _find_data_wide_csv(self, experiment: str, split: str) -> Optional[Path]:
        """Legacy method - returns first file or None."""
        files = self._find_data_wide_csvs(experiment, split)
        return files[0] if files else None

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        for exp in self._experiments:
            if self._find_data_wide_csv(exp, self.split) is None:
                return False
        return True

    @staticmethod
    def _safe_extract_zip(zip_path: Path, dst_root: Path) -> None:
        """Safely extract zip into dst_root (prevents path traversal/zip-slip)."""
        dst_root = Path(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)
        root_resolved = dst_root.resolve()

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                target_path = (dst_root / member_path).resolve()

                if root_resolved != target_path and root_resolved not in target_path.parents:
                    raise RuntimeError(f"Unsafe path in zip: {member.filename}")

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, 'r') as src, open(target_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

    def _download(self) -> None:
        """Download and extract BBIL experiment archive from GitHub releases."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.ZIP_FILENAME

        if not zip_path.exists():
            print("Downloading BBIL experiment archive from GitHub releases...")
            try:
                download_url(
                    url=self.DOWNLOAD_URL,
                    root=self.data_root,
                    filename=self.ZIP_FILENAME,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download BBIL dataset: {e}\n"
                    f"Please download manually from: {self.DOWNLOAD_URL}"
                )

        print("Extracting dataset files...")
        try:
            self._safe_extract_zip(zip_path, self.data_root)
        except Exception as e:
            raise RuntimeError(f"Failed to extract BBIL dataset: {e}")

        if not self._check_exists():
            raise RuntimeError(
                "BBIL archive was downloaded/extracted, but expected files were not found.\n"
                "Expected structure like: experiment{1,2}/{train,valid,test}/*_data_wide.csv"
            )

    def _load_data(self) -> None:
        """Load BBIL dataset from data_wide.csv files (predefined splits).

        BBIL has multiple *_data_wide.csv files per split (one per recording session).
        This method loads and concatenates all files.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load BBIL dataset.\n"
                "Install with: pip install pandas"
            )

        # Collect all CSV files for each experiment
        csv_paths: Dict[str, List[Path]] = {}
        for exp in self._experiments:
            files = self._find_data_wide_csvs(exp, self.split)
            if not files:
                raise FileNotFoundError(f"No *_data_wide.csv found for {exp}/{self.split} under {self.data_root}")
            csv_paths[exp] = files

        # Discover all RSSI/edge columns from all files
        # BBIL uses "edge_N" column names (e.g., edge_1, edge_2, ...)
        rssi_cols_set = set()
        for files in csv_paths.values():
            for p in files:
                header = pd.read_csv(p, nrows=0)
                rssi_cols_set.update([c for c in header.columns if self._EDGE_COL_RE.match(str(c))])

        if not rssi_cols_set:
            raise ValueError("No edge_{N} columns found in BBIL data_wide.csv")

        def sort_key(col: str):
            m = self._EDGE_COL_RE.match(str(col))
            if m:
                return (0, int(m.group(1)), str(col).lower())
            m2 = re.search(r'(\d+)$', str(col))
            if m2:
                return (1, int(m2.group(1)), str(col).lower())
            return (2, 0, str(col).lower())

        rssi_cols = sorted(rssi_cols_set, key=sort_key)

        def edge_id(col: str) -> str:
            m = self._EDGE_COL_RE.match(str(col))
            if m:
                return f"edge{int(m.group(1))}"
            return str(col)

        beacon_ids = [edge_id(c) for c in rssi_cols]
        self._beacon_ids = beacon_ids
        self._num_beacons = len(beacon_ids)
        col_to_idx = {c: i for i, c in enumerate(rssi_cols)}

        def find_column(df, candidates: List[str]) -> Optional[str]:
            cols = list(df.columns)
            lower = {str(c).lower(): str(c) for c in cols}
            for cand in candidates:
                if cand.lower() in lower:
                    return lower[cand.lower()]
            return None

        total_rows = 0
        for exp, csv_files in csv_paths.items():
            floor_num = self.EXPERIMENT_TO_FLOOR[exp]

            # Load and process each CSV file for this experiment
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)

                x_col = find_column(df, ['realx'])
                y_col = find_column(df, ['realy'])
                if x_col is None or y_col is None:
                    raise ValueError(f"Expected columns realx/realy not found in {csv_path}")

                xs = pd.to_numeric(df[x_col], errors='coerce')
                ys = pd.to_numeric(df[y_col], errors='coerce')
                valid = xs.notna() & ys.notna()
                if not bool(valid.all()):
                    df = df.loc[valid].reset_index(drop=True)
                    xs = xs.loc[valid].reset_index(drop=True)
                    ys = ys.loc[valid].reset_index(drop=True)

                num_samples = len(df)
                rssi_mat = np.full((num_samples, len(rssi_cols)), self.NOT_DETECTED_VALUE, dtype=np.float32)

                present_cols = [c for c in rssi_cols if c in df.columns]
                if present_cols:
                    values = df[present_cols].to_numpy(dtype=np.float32, copy=True)
                    values = np.where(np.isnan(values), self.NOT_DETECTED_VALUE, values)
                    values = np.clip(values, self.NOT_DETECTED_VALUE, 0.0)
                    for j, col in enumerate(present_cols):
                        rssi_mat[:, col_to_idx[col]] = values[:, j]

                dt_col = find_column(df, ['datetime'])
                beaconid_col = find_column(df, ['beaconid'])

                for i in range(num_samples):
                    signal = BLESignal(
                        rssi_values=rssi_mat[i].copy(),
                        beacon_ids=beacon_ids,
                    )
                    location = Location(
                        coordinate=Coordinate(x=float(xs.iloc[i]), y=float(ys.iloc[i])),
                        floor=floor_num,
                        building_id='0',
                    )

                    # Store metadata
                    meta: Dict[str, Any] = {
                        'experiment': exp,
                        'split': self.split,
                        'source_file': csv_path.name,
                    }
                    if dt_col is not None:
                        meta['datetime'] = str(df.at[i, dt_col])
                    if beaconid_col is not None:
                        meta['beaconid'] = df.at[i, beaconid_col]

                    self._signals.append(signal)
                    self._locations.append(location)
                    self._metadata.append(meta)

                total_rows += num_samples

        print(f"Loaded {len(self._signals)} samples from BBIL ({self.split})")
        print(f"Feature dimension (edges): {self._num_beacons}")
        print(f"Experiments: {self._experiments}")



def BLEIndoor(data_root=None, split=None, download=False, floor='all', experiment='all', **kwargs):
    """
    Convenience function for loading BLEIndoor dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'valid'/'val', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        floor: Legacy alias for experiment selection (1->experiment1, 2->experiment2, 'all'->both)
        experiment: Experiment(s) to load ('experiment1', 'experiment2', 'all', or list)
        **kwargs: Additional arguments passed to BLEIndoorDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> train, test = BLEIndoor(download=True)
        >>> train = BLEIndoor(experiment='experiment1', split='train')
        >>> BLEIndoor.list_experiments()
    """
    if split is None:
        train_dataset = BLEIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            floor=floor,
            experiment=experiment,
            **kwargs
        )
        test_dataset = BLEIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            floor=floor,
            experiment=experiment,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = BLEIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            floor=floor,
            experiment=experiment,
            **kwargs
        )
        test_dataset = BLEIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            floor=floor,
            experiment=experiment,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return BLEIndoorDataset(
            data_root=data_root,
            split=split,
            download=download,
            floor=floor,
            experiment=experiment,
            **kwargs
        )


BLEIndoor.list_floors = BLEIndoorDataset.list_floors
BLEIndoor.list_experiments = BLEIndoorDataset.list_experiments
