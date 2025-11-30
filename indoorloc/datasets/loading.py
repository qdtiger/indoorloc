"""
Dataset Loading Functions

HuggingFace-style dataset loading for IndoorLoc.

Example:
    >>> import indoorloc as iloc
    >>> train, test = iloc.load_dataset("ujindoorloc")
    >>> train = iloc.load_dataset("ujindoorloc", split="train")
"""
from typing import Union, Tuple, Optional, List

from .catalog import get_dataset_class_name, get_all_dataset_names, DATASETS_BY_SIGNAL
from ..registry import DATASETS


def load_dataset(
    name: str,
    split: Optional[str] = None,
    download: bool = True,
    **kwargs
) -> Union['BaseDataset', Tuple['BaseDataset', 'BaseDataset']]:
    """
    Load a dataset by short name (HuggingFace-style).

    This is the recommended way to load datasets in IndoorLoc. It provides
    a simple, consistent interface for loading any supported dataset.

    Args:
        name: Dataset short name (e.g., 'ujindoorloc', 'tampere', 'ble_indoor')
            Use list_datasets() to see all available names.
        split: Which split to load:
            - None (default): Returns (train, test) tuple
            - 'train': Returns only training set
            - 'test': Returns only test set
        download: Whether to download the dataset if not found locally.
        **kwargs: Additional dataset-specific arguments (e.g., building='0')

    Returns:
        - If split=None: Tuple of (train_dataset, test_dataset)
        - If split='train' or 'test': Single dataset

    Example:
        >>> import indoorloc as iloc
        >>>
        >>> # Load both train and test (recommended)
        >>> train, test = iloc.load_dataset("ujindoorloc")
        >>>
        >>> # Load single split
        >>> train = iloc.load_dataset("ujindoorloc", split="train")
        >>>
        >>> # With options
        >>> train, test = iloc.load_dataset("ujindoorloc", download=True)
        >>>
        >>> # Dataset-specific options
        >>> train, test = iloc.load_dataset("ujindoorloc", building='0')

    Raises:
        ValueError: If dataset name is not found
        FileNotFoundError: If download=False and data not found locally
    """
    # Resolve short name to registry class name
    class_name = get_dataset_class_name(name)

    # Get the dataset class from registry
    dataset_cls = DATASETS.get(class_name)
    if dataset_cls is None:
        raise ValueError(
            f"Dataset '{class_name}' is registered in catalog but not in DATASETS registry.\n"
            f"This is an internal error. Please report this issue."
        )

    # Build common kwargs
    common_kwargs = {'download': download, **kwargs}

    if split is not None:
        # Return single split
        if split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        return dataset_cls(split=split, **common_kwargs)
    else:
        # Return (train, test) tuple
        train_dataset = dataset_cls(split='train', **common_kwargs)
        test_dataset = dataset_cls(split='test', **common_kwargs)
        return train_dataset, test_dataset


def list_datasets(signal_type: Optional[str] = None) -> List[str]:
    """
    List available dataset names.

    Args:
        signal_type: Filter by signal type. Options:
            - None: Return all datasets
            - 'wifi': WiFi RSSI datasets
            - 'ble': Bluetooth Low Energy datasets
            - 'csi': Channel State Information datasets
            - 'uwb': Ultra-Wideband datasets
            - 'hybrid': Multi-modal/hybrid datasets
            - 'magnetic': Magnetic field datasets
            - 'other': Other signal types (VLC, ultrasound, RFID, etc.)

    Returns:
        List of dataset short names

    Example:
        >>> iloc.list_datasets()
        ['ble_indoor', 'ble_rssi_uci', 'csi2pos', 'csi2taoa', ...]
        >>> iloc.list_datasets('wifi')
        ['longtermwifi', 'sodindoorloc', 'tampere', 'tuji1', 'ujindoorloc', 'wlanrssi']
    """
    if signal_type is None:
        return get_all_dataset_names()

    signal_type_lower = signal_type.lower()
    if signal_type_lower not in DATASETS_BY_SIGNAL:
        available = list(DATASETS_BY_SIGNAL.keys())
        raise ValueError(
            f"Unknown signal type: '{signal_type}'\n"
            f"Available types: {available}"
        )

    return sorted(DATASETS_BY_SIGNAL[signal_type_lower])


def dataset_info(name: str) -> dict:
    """
    Get information about a dataset.

    Args:
        name: Dataset short name

    Returns:
        Dictionary with dataset information:
            - class_name: Registry class name
            - signal_type: Primary signal type
            - description: Dataset description (from docstring)

    Example:
        >>> info = iloc.dataset_info("ujindoorloc")
        >>> print(info['signal_type'])
        'wifi'
    """
    class_name = get_dataset_class_name(name)
    dataset_cls = DATASETS.get(class_name)

    # Find signal type
    signal_type = 'unknown'
    for sig_type, datasets in DATASETS_BY_SIGNAL.items():
        if name.lower() in datasets:
            signal_type = sig_type
            break

    # Get description from docstring
    description = ""
    if dataset_cls is not None and dataset_cls.__doc__:
        # Get first paragraph of docstring
        doc_lines = dataset_cls.__doc__.strip().split('\n')
        description = doc_lines[0].strip() if doc_lines else ""

    return {
        'name': name,
        'class_name': class_name,
        'signal_type': signal_type,
        'description': description,
    }


__all__ = [
    'load_dataset',
    'list_datasets',
    'dataset_info',
]
