"""
Core Transform Classes for Indoor Localization

Provides base transform class and composition utilities.
"""
from abc import ABC, abstractmethod
from typing import List, Union, Any

from ...registry import TRANSFORMS


class BaseTransform(ABC):
    """Abstract base class for all transforms.

    All transform implementations should inherit from this class
    and implement the __call__ method.

    Example:
        >>> class MyTransform(BaseTransform):
        ...     def __call__(self, signal):
        ...         # Transform logic here
        ...         return transformed_signal
    """

    @abstractmethod
    def __call__(self, signal: Any) -> Any:
        """Apply the transform to a signal.

        Args:
            signal: Input signal to transform.

        Returns:
            Transformed signal.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of transforms or config dicts to compose.

    Example:
        >>> transform = Compose([
        ...     APFilter(threshold=-90),
        ...     RSSINormalize(method='minmax'),
        ... ])
        >>> transformed_signal = transform(signal)
    """

    def __init__(self, transforms: List[Union['BaseTransform', dict]]):
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                # Build transform from config dict
                self.transforms.append(TRANSFORMS.build(t))
            else:
                self.transforms.append(t)

    def __call__(self, signal: Any) -> Any:
        """Apply all transforms sequentially.

        Args:
            signal: Input signal.

        Returns:
            Transformed signal after applying all transforms.
        """
        for transform in self.transforms:
            signal = transform(signal)
        return signal

    def __repr__(self) -> str:
        transforms_str = ', '.join(repr(t) for t in self.transforms)
        return f"Compose([{transforms_str}])"

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int) -> 'BaseTransform':
        return self.transforms[idx]


@TRANSFORMS.register_module()
class Identity(BaseTransform):
    """Identity transform that returns the input unchanged.

    Useful as a placeholder or for conditional pipelines.
    """

    def __call__(self, signal: Any) -> Any:
        return signal
