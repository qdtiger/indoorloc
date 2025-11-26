"""
Data Loaders for Various File Formats

Provides utilities for loading data from HDF5, MATLAB, and other formats
commonly used in indoor localization datasets.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

# Optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_hdf5(
    filepath: Union[str, Path],
    keys: Optional[list] = None
) -> Dict[str, Any]:
    """Load data from HDF5 file.

    Args:
        filepath: Path to HDF5 file.
        keys: Optional list of specific keys to load (None for all).

    Returns:
        Dictionary mapping keys to loaded data arrays.

    Raises:
        ImportError: If h5py is not installed.
        FileNotFoundError: If file does not exist.

    Example:
        >>> # Load specific datasets from HDF5
        >>> data = load_hdf5('dataset.h5', keys=['rssi', 'locations'])
        >>> rssi = data['rssi']
        >>> locations = data['locations']
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for loading HDF5 files.\n"
            "Install with: pip install indoorloc[datasets]\n"
            "Or: pip install h5py"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    result = {}

    with h5py.File(filepath, 'r') as f:
        # If no keys specified, load all datasets
        if keys is None:
            keys = list(f.keys())

        for key in keys:
            if key not in f:
                print(f"Warning: Key '{key}' not found in HDF5 file")
                continue

            # Load dataset
            dataset = f[key]

            # Handle different dataset types
            if isinstance(dataset, h5py.Dataset):
                result[key] = dataset[()]
            elif isinstance(dataset, h5py.Group):
                # Recursively load group contents
                result[key] = _load_hdf5_group(dataset)

    return result


def _load_hdf5_group(group: 'h5py.Group') -> Dict[str, Any]:
    """Recursively load HDF5 group contents.

    Args:
        group: HDF5 group object.

    Returns:
        Dictionary of group contents.
    """
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = _load_hdf5_group(item)
    return result


def load_mat(
    filepath: Union[str, Path],
    keys: Optional[list] = None,
    squeeze_me: bool = True,
    struct_as_record: bool = False
) -> Dict[str, Any]:
    """Load data from MATLAB .mat file.

    Args:
        filepath: Path to .mat file.
        keys: Optional list of specific keys to load (None for all).
        squeeze_me: Whether to squeeze unit matrix dimensions.
        struct_as_record: Whether to return structs as record arrays.

    Returns:
        Dictionary mapping variable names to loaded data.

    Raises:
        ImportError: If scipy is not installed.
        FileNotFoundError: If file does not exist.

    Example:
        >>> # Load MATLAB file
        >>> data = load_mat('dataset.mat')
        >>> rssi = data['rssi']
        >>> coords = data['coordinates']
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for loading MATLAB .mat files.\n"
            "Install with: pip install indoorloc[datasets]\n"
            "Or: pip install scipy"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")

    # Load MATLAB file
    mat_data = loadmat(
        filepath,
        squeeze_me=squeeze_me,
        struct_as_record=struct_as_record
    )

    # Filter out MATLAB metadata (keys starting with '__')
    result = {
        k: v for k, v in mat_data.items()
        if not k.startswith('__')
    }

    # If specific keys requested, filter result
    if keys is not None:
        result = {k: result[k] for k in keys if k in result}

    return result


def save_hdf5(
    filepath: Union[str, Path],
    data: Dict[str, np.ndarray],
    compression: Optional[str] = 'gzip'
) -> None:
    """Save data to HDF5 file.

    Args:
        filepath: Path to save HDF5 file.
        data: Dictionary mapping keys to numpy arrays.
        compression: Compression method ('gzip', 'lzf', None).

    Raises:
        ImportError: If h5py is not installed.

    Example:
        >>> # Save data to HDF5
        >>> data = {'rssi': rssi_array, 'locations': loc_array}
        >>> save_hdf5('dataset.h5', data)
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for saving HDF5 files.\n"
            "Install with: pip install indoorloc[datasets]\n"
            "Or: pip install h5py"
        )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        for key, value in data.items():
            if isinstance(value, dict):
                # Create group for nested dict
                group = f.create_group(key)
                _save_hdf5_group(group, value, compression)
            else:
                # Create dataset
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression
                )


def _save_hdf5_group(
    group: 'h5py.Group',
    data: Dict[str, Any],
    compression: Optional[str]
) -> None:
    """Recursively save nested dictionary to HDF5 group.

    Args:
        group: HDF5 group object.
        data: Dictionary to save.
        compression: Compression method.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _save_hdf5_group(subgroup, value, compression)
        else:
            group.create_dataset(
                key,
                data=value,
                compression=compression
            )


def load_csv_with_header(
    filepath: Union[str, Path],
    delimiter: str = ',',
    skip_rows: int = 0
) -> Dict[str, np.ndarray]:
    """Load CSV file with header into dictionary of arrays.

    Args:
        filepath: Path to CSV file.
        delimiter: Column delimiter.
        skip_rows: Number of rows to skip before header.

    Returns:
        Dictionary mapping column names to arrays.

    Raises:
        FileNotFoundError: If file does not exist.

    Example:
        >>> # Load CSV with header
        >>> data = load_csv_with_header('data.csv')
        >>> rssi = data['rssi_AP1']
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Load data with numpy
    with open(filepath, 'r') as f:
        # Skip initial rows if needed
        for _ in range(skip_rows):
            next(f)

        # Read header
        header = next(f).strip().split(delimiter)

    # Load data (skip header + skip_rows)
    data_array = np.loadtxt(
        filepath,
        delimiter=delimiter,
        skiprows=skip_rows + 1
    )

    # Create dictionary mapping column names to data
    result = {}
    for i, col_name in enumerate(header):
        result[col_name] = data_array[:, i]

    return result


def check_optional_dependencies() -> Dict[str, bool]:
    """Check which optional data loading dependencies are installed.

    Returns:
        Dictionary mapping dependency names to availability status.

    Example:
        >>> deps = check_optional_dependencies()
        >>> if not deps['h5py']:
        ...     print("HDF5 support not available")
    """
    return {
        'h5py': HAS_H5PY,
        'scipy': HAS_SCIPY,
    }


def print_dependency_status() -> None:
    """Print status of optional data loading dependencies."""
    deps = check_optional_dependencies()

    print("Optional Data Loading Dependencies:")
    print("-" * 40)

    for name, available in deps.items():
        status = "✓ Installed" if available else "✗ Not installed"
        print(f"  {name:10} {status}")

    print()

    if not all(deps.values()):
        print("To install all optional dependencies:")
        print("  pip install indoorloc[datasets]")


__all__ = [
    'load_hdf5',
    'load_mat',
    'save_hdf5',
    'load_csv_with_header',
    'check_optional_dependencies',
    'print_dependency_status',
]
