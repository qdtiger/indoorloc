"""
Base Localizer Classes

Provides abstract base classes for all localization algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import time
import joblib

from ..signals.base import BaseSignal
from ..locations.location import Location, LocalizationResult
from ..registry import LOCALIZERS


class BaseLocalizer(ABC):
    """
    Abstract base class for all localization algorithms.

    All localization implementations (k-NN, SVM, CNN, etc.) should inherit
    from this class and implement the abstract methods.

    This provides a unified interface for:
    - Traditional ML methods (k-NN, SVM, RF)
    - Deep learning methods (CNN, LSTM, Transformer)
    - Fusion methods (Kalman, Particle Filter)

    Attributes:
        input_signals: List of expected input signal types
        output_type: Type of output ('regression', 'classification', 'hybrid')
        _is_trained: Whether the model has been trained
    """

    def __init__(
        self,
        input_signals: Optional[List[str]] = None,
        output_type: str = 'hybrid',
        **kwargs
    ):
        """
        Initialize localizer.

        Args:
            input_signals: List of input signal types (e.g., ['wifi', 'imu'])
            output_type: Output type
                - 'regression': Coordinate prediction only
                - 'classification': Floor/building/room prediction only
                - 'hybrid': Both coordinate and classification
        """
        self.input_signals = input_signals or ['wifi']
        self.output_type = output_type
        self._is_trained = False
        self._config = kwargs

    @property
    @abstractmethod
    def localizer_type(self) -> str:
        """
        Get the localizer type identifier.

        Returns:
            String identifier (e.g., 'knn', 'svm', 'cnn')
        """
        pass

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained

    @abstractmethod
    def fit(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        **kwargs
    ) -> 'BaseLocalizer':
        """
        Train/fit the localizer.

        Args:
            signals: List of training signals
            locations: List of corresponding ground truth locations
            **kwargs: Additional training arguments

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, signal: BaseSignal) -> LocalizationResult:
        """
        Predict location for a single signal.

        Args:
            signal: Input signal

        Returns:
            Localization result with predicted location
        """
        pass

    def predict_batch(
        self,
        signals: List[BaseSignal]
    ) -> List[LocalizationResult]:
        """
        Predict locations for multiple signals.

        Default implementation calls predict() for each signal.
        Subclasses can override for optimized batch prediction.

        Args:
            signals: List of input signals

        Returns:
            List of localization results
        """
        return [self.predict(s) for s in signals]

    def predict_timed(self, signal: BaseSignal) -> LocalizationResult:
        """
        Predict with timing information.

        Args:
            signal: Input signal

        Returns:
            Localization result with latency_ms populated
        """
        start = time.perf_counter()
        result = self.predict(signal)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        result.latency_ms = elapsed
        return result

    def save(self, path: str) -> None:
        """
        Save the localizer to file.

        Default implementation uses joblib.
        Subclasses (especially deep learning) should override.

        Args:
            path: File path to save to
        """
        save_dict = {
            'localizer_type': self.localizer_type,
            'input_signals': self.input_signals,
            'output_type': self.output_type,
            'is_trained': self._is_trained,
            'config': self._config,
            'state': self._get_state()
        }
        joblib.dump(save_dict, path)

    def load(self, path: str) -> 'BaseLocalizer':
        """
        Load the localizer from file.

        Args:
            path: File path to load from

        Returns:
            Self for method chaining
        """
        save_dict = joblib.load(path)
        self.input_signals = save_dict['input_signals']
        self.output_type = save_dict['output_type']
        self._is_trained = save_dict['is_trained']
        self._config = save_dict['config']
        self._set_state(save_dict['state'])
        return self

    def _get_state(self) -> Dict[str, Any]:
        """
        Get internal state for saving.

        Subclasses should override to save model-specific state.

        Returns:
            State dictionary
        """
        return {}

    def _set_state(self, state: Dict[str, Any]) -> None:
        """
        Set internal state from loading.

        Subclasses should override to load model-specific state.

        Args:
            state: State dictionary
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get localizer configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'type': self.localizer_type,
            'input_signals': self.input_signals,
            'output_type': self.output_type,
            **self._config
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.localizer_type}, "
            f"trained={self._is_trained})"
        )


class TraditionalLocalizer(BaseLocalizer):
    """
    Base class for traditional ML localizers (k-NN, SVM, RF).

    Provides common functionality for sklearn-based models.
    """

    def __init__(
        self,
        predict_floor: bool = True,
        predict_building: bool = True,
        **kwargs
    ):
        """
        Initialize traditional localizer.

        Args:
            predict_floor: Whether to predict floor
            predict_building: Whether to predict building
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.predict_floor = predict_floor
        self.predict_building = predict_building

        # Models will be set by subclasses
        self._coord_model = None
        self._floor_model = None
        self._building_model = None

    def _extract_features(self, signal: BaseSignal) -> 'np.ndarray':
        """
        Extract feature vector from signal.

        Args:
            signal: Input signal

        Returns:
            Feature vector as numpy array
        """
        return signal.to_numpy()

    def _extract_labels(
        self,
        locations: List[Location]
    ) -> Dict[str, 'np.ndarray']:
        """
        Extract labels from locations.

        Args:
            locations: List of locations

        Returns:
            Dictionary with 'coords', 'floor', 'building' arrays
        """
        import numpy as np

        coords = np.array([
            [loc.coordinate.x, loc.coordinate.y]
            for loc in locations
        ], dtype=np.float32)

        floors = np.array([
            loc.floor if loc.floor is not None else -1
            for loc in locations
        ], dtype=np.int32)

        buildings = np.array([
            int(loc.building_id) if loc.building_id is not None else -1
            for loc in locations
        ], dtype=np.int32)

        return {
            'coords': coords,
            'floor': floors,
            'building': buildings
        }


__all__ = ['BaseLocalizer', 'TraditionalLocalizer']
