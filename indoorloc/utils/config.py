"""
Configuration System

YAML-based configuration with inheritance support.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import copy
import inspect


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

    def validate(self) -> List[str]:
        """
        Validate configuration completeness.

        Returns:
            List of warning messages (empty if valid)

        Example:
            >>> cfg = Config.fromfile('config.yaml')
            >>> warnings = cfg.validate()
            >>> if warnings:
            ...     print("Warnings:", warnings)
        """
        warnings = []

        # Check for required sections
        if 'model' not in self:
            warnings.append("Missing 'model' section")
        elif isinstance(self.model, Config):
            if 'type' not in self.model._cfg_dict:
                warnings.append("Missing 'model.type' field")

        if 'dataset' not in self:
            warnings.append("Missing 'dataset' section (optional for inference)")

        # Check deep learning specific requirements
        if 'model' in self and isinstance(self.model, Config):
            model_dict = self.model._cfg_dict
            if model_dict.get('type') == 'DeepLocalizer':
                if 'backbone' not in model_dict:
                    warnings.append("DeepLocalizer requires 'model.backbone'")
                if 'head' not in model_dict:
                    warnings.append("DeepLocalizer requires 'model.head'")

        return warnings

    def keys(self):
        """Return config keys."""
        return self._cfg_dict.keys()

    def items(self):
        """Return config items."""
        return self._cfg_dict.items()

    def values(self):
        """Return config values."""
        return self._cfg_dict.values()


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


def print_config_help(module_type: str, show_docstring: bool = True):
    """
    Print available parameters for a model/module type.

    Args:
        module_type: Type name (e.g., 'KNNLocalizer', 'TimmBackbone', 'RegressionHead')
        show_docstring: Whether to show class docstring

    Example:
        >>> print_config_help('KNNLocalizer')
        >>> print_config_help('TimmBackbone')
        >>> print_config_help('RegressionHead')
    """
    # Import registries
    try:
        from ..registry import LOCALIZERS, BACKBONES, HEADS
    except ImportError:
        print("Error: Registry not available. Make sure indoorloc is properly installed.")
        return

    # Find the class in registries
    cls = None
    registry_name = None

    for name, registry in [('LOCALIZERS', LOCALIZERS), ('BACKBONES', BACKBONES), ('HEADS', HEADS)]:
        try:
            cls = registry.get(module_type)
            if cls is not None:
                registry_name = name
                break
        except KeyError:
            continue

    if cls is None:
        print(f"Unknown module type: {module_type}")
        print("\nAvailable types:")
        print("  LOCALIZERS:", ', '.join(LOCALIZERS.list_modules()))
        print("  BACKBONES:", ', '.join(BACKBONES.list_modules()))
        print("  HEADS:", ', '.join(HEADS.list_modules()))
        return

    print(f"\n{module_type} ({registry_name})")
    print("=" * 60)

    # Show docstring
    if show_docstring and cls.__doc__:
        doc_lines = cls.__doc__.strip().split('\n')
        print(doc_lines[0])
        print()

    # Show parameters
    print("Parameters:")
    print("-" * 60)

    try:
        sig = inspect.signature(cls.__init__)
        for name, param in sig.parameters.items():
            if name in ('self', 'args', 'kwargs'):
                continue

            # Get default value
            if param.default is inspect.Parameter.empty:
                default = 'required'
                default_str = f"  {name}: {default}"
            else:
                default = param.default
                default_str = f"  {name}: {repr(default)}"

            # Get type annotation
            if param.annotation is not inspect.Parameter.empty:
                try:
                    type_name = param.annotation.__name__
                except AttributeError:
                    type_name = str(param.annotation)
                default_str += f"  ({type_name})"

            print(default_str)
    except (ValueError, TypeError) as e:
        print(f"  Unable to inspect parameters: {e}")

    print()


def get_default_config(module_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a module type.

    Args:
        module_type: Type name (e.g., 'KNNLocalizer', 'TimmBackbone')

    Returns:
        Dictionary of default parameter values

    Example:
        >>> defaults = get_default_config('KNNLocalizer')
        >>> print(defaults)
        {'k': 5, 'weights': 'distance', ...}
    """
    try:
        from ..registry import LOCALIZERS, BACKBONES, HEADS
    except ImportError:
        return {}

    # Find the class
    cls = None
    for registry in [LOCALIZERS, BACKBONES, HEADS]:
        try:
            cls = registry.get(module_type)
            if cls is not None:
                break
        except KeyError:
            continue

    if cls is None:
        return {}

    # Extract defaults from signature
    defaults = {'type': module_type}
    try:
        sig = inspect.signature(cls.__init__)
        for name, param in sig.parameters.items():
            if name in ('self', 'args', 'kwargs'):
                continue
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default
    except (ValueError, TypeError):
        pass

    return defaults


__all__ = ['Config', 'load_config', 'merge_configs', 'print_config_help', 'get_default_config']
