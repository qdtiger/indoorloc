"""
IndoorLoc Utilities Module

Provides utility functions and classes.
"""
from .config import Config, load_config, merge_configs, print_config_help, get_default_config
from .download import (
    get_data_home,
    check_integrity,
    download_url,
    download_and_extract_zip,
    download_from_zenodo,
    download_from_uci,
)
from .data_loaders import (
    load_hdf5,
    load_mat,
    save_hdf5,
    load_csv_with_header,
    check_optional_dependencies,
    print_dependency_status,
)
from .explain import (
    explain_model,
    explain_dataset,
    explain_config,
)

__all__ = [
    'Config',
    'load_config',
    'merge_configs',
    'print_config_help',
    'get_default_config',
    'get_data_home',
    'check_integrity',
    'download_url',
    'download_and_extract_zip',
    'download_from_zenodo',
    'download_from_uci',
    'load_hdf5',
    'load_mat',
    'save_hdf5',
    'load_csv_with_header',
    'check_optional_dependencies',
    'print_dependency_status',
    'explain_model',
    'explain_dataset',
    'explain_config',
]
