"""
IndoorLoc Registry System

A simplified registry system for managing configurable modules.
Inspired by MMEngine but lighter weight.
"""
from typing import Dict, Type, Any, Optional, Callable, List
import inspect


class Registry:
    """
    Module Registry for managing configurable components.

    Example:
        >>> LOCALIZERS = Registry('localizers')
        >>> @LOCALIZERS.register_module()
        ... class KNNLocalizer:
        ...     pass
        >>> model = LOCALIZERS.build({'type': 'KNNLocalizer', 'k': 5})
    """

    def __init__(self, name: str, parent: Optional['Registry'] = None):
        """
        Initialize a registry.

        Args:
            name: Registry name for identification
            parent: Parent registry for hierarchical lookups
        """
        self._name = name
        self._parent = parent
        self._module_dict: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        """Get all registered modules."""
        return self._module_dict.copy()

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Type] = None
    ) -> Callable:
        """
        Register a module.

        Can be used as a decorator or called directly.

        Args:
            name: Module name. If None, use class name.
            force: Force registration even if name exists.
            module: Module class to register directly.

        Returns:
            Decorator function or the module itself.

        Example:
            >>> @LOCALIZERS.register_module()
            ... class KNNLocalizer:
            ...     pass

            >>> @LOCALIZERS.register_module(name='knn')
            ... class KNNLocalizer:
            ...     pass
        """
        if module is not None:
            self._register(module, name, force)
            return module

        def decorator(cls: Type) -> Type:
            self._register(cls, name, force)
            return cls

        return decorator

    def _register(
        self,
        module: Type,
        name: Optional[str] = None,
        force: bool = False
    ) -> None:
        """Internal registration method."""
        if name is None:
            name = module.__name__

        if not force and name in self._module_dict:
            raise KeyError(
                f"'{name}' is already registered in '{self._name}' registry. "
                f"Use force=True to override."
            )

        self._module_dict[name] = module

    def get(self, name: str) -> Optional[Type]:
        """
        Get a registered module by name.

        Args:
            name: Module name.

        Returns:
            Module class or None if not found.
        """
        if name in self._module_dict:
            return self._module_dict[name]
        if self._parent is not None:
            return self._parent.get(name)
        return None

    def build(self, cfg: Dict[str, Any]) -> Any:
        """
        Build a module from config dict.

        Args:
            cfg: Config dict with 'type' key specifying module name.

        Returns:
            Instantiated module.

        Example:
            >>> model = LOCALIZERS.build({'type': 'KNNLocalizer', 'k': 5})
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a dict, got {type(cfg)}")

        if 'type' not in cfg:
            raise KeyError("Config must contain 'type' key")

        cfg = cfg.copy()
        module_type = cfg.pop('type')

        module_cls = self.get(module_type)
        if module_cls is None:
            raise KeyError(
                f"'{module_type}' is not registered in '{self._name}' registry. "
                f"Available: {list(self._module_dict.keys())}"
            )

        return module_cls(**cfg)

    def list_modules(self) -> List[str]:
        """List all registered module names."""
        return list(self._module_dict.keys())


# ==============================================================================
# Global Registries (10 total)
# ==============================================================================

# Signal types (WiFi, BLE, IMU, etc.)
SIGNALS = Registry('signals')

# Datasets (UJIndoorLoc, Tampere, etc.)
DATASETS = Registry('datasets')

# Data transforms (normalize, augment, etc.)
TRANSFORMS = Registry('transforms')

# Localization algorithms (k-NN, SVM, CNN, etc.)
LOCALIZERS = Registry('localizers')

# Fusion algorithms (Kalman, Particle Filter, etc.)
FUSIONS = Registry('fusions')

# Evaluation metrics (MeanError, FloorAccuracy, etc.)
METRICS = Registry('metrics')

# Neural network backbones (for deep learning)
BACKBONES = Registry('backbones')

# Prediction heads (regression, classification)
HEADS = Registry('heads')

# Trainers (for model training)
TRAINERS = Registry('trainers')

# Visualizers (trajectory, error heatmap, etc.)
VISUALIZERS = Registry('visualizers')


__all__ = [
    'Registry',
    'SIGNALS',
    'DATASETS',
    'TRANSFORMS',
    'LOCALIZERS',
    'FUSIONS',
    'METRICS',
    'BACKBONES',
    'HEADS',
    'TRAINERS',
    'VISUALIZERS',
]
