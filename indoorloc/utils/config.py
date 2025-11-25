"""
Configuration System

YAML-based configuration with inheritance support.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import copy


class Config:
    """
    Configuration class with YAML loading and inheritance support.

    Supports:
    - YAML file loading
    - Configuration inheritance via _base_ key
    - Deep merge of nested configs
    - Attribute-style access

    Example:
        >>> cfg = Config.fromfile('configs/wifi/knn_ujindoorloc.yaml')
        >>> print(cfg.model.type)
        'KNNLocalizer'
        >>> print(cfg.model.k)
        5
    """

    def __init__(self, cfg_dict: Dict[str, Any] = None):
        """
        Initialize config from dictionary.

        Args:
            cfg_dict: Configuration dictionary
        """
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict

        # Convert nested dicts to Config objects for attribute access
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def fromfile(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file with inheritance support.

        Args:
            filepath: Path to YAML config file

        Returns:
            Config instance

        Example:
            >>> cfg = Config.fromfile('configs/wifi/knn_ujindoorloc.yaml')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f) or {}

        # Handle inheritance
        if '_base_' in cfg_dict:
            cfg_dict = cls._merge_base_configs(cfg_dict, filepath.parent)

        return cls(cfg_dict)

    @classmethod
    def _merge_base_configs(
        cls,
        cfg_dict: Dict[str, Any],
        base_dir: Path
    ) -> Dict[str, Any]:
        """
        Merge base configurations.

        Args:
            cfg_dict: Current config with _base_ key
            base_dir: Directory to resolve relative paths

        Returns:
            Merged configuration dictionary
        """
        base_files = cfg_dict.pop('_base_')

        if isinstance(base_files, str):
            base_files = [base_files]

        # Load and merge base configs
        merged = {}
        for base_file in base_files:
            base_path = base_dir / base_file
            base_cfg = cls.fromfile(base_path)
            merged = cls._deep_merge(merged, base_cfg.to_dict())

        # Merge current config (override base)
        merged = cls._deep_merge(merged, cfg_dict)

        return merged

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)
            ):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        result = {}
        for key, value in self._cfg_dict.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._cfg_dict.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._cfg_dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._cfg_dict

    def __repr__(self) -> str:
        return f"Config({self._cfg_dict})"

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)


def load_config(filepath: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Convenience function for Config.fromfile().

    Args:
        filepath: Path to YAML config file

    Returns:
        Config instance

    Example:
        >>> cfg = load_config('configs/wifi/knn_ujindoorloc.yaml')
    """
    return Config.fromfile(filepath)


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configurations.

    Later configs override earlier ones.

    Args:
        *configs: Config instances to merge

    Returns:
        Merged Config instance
    """
    merged = {}
    for cfg in configs:
        merged = Config._deep_merge(merged, cfg.to_dict())
    return Config(merged)


__all__ = ['Config', 'load_config', 'merge_configs']
