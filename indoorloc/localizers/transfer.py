"""
Transfer Learning Localizers

Domain adaptation methods for cross-device, cross-time, cross-environment localization.
Based on SKADA (scikit-adaptation) library.
"""
from typing import List, Optional, Dict, Any, Union

import numpy as np

from .base import BaseLocalizer, TraditionalLocalizer
from ..signals.base import BaseSignal
from ..locations.location import Location, LocalizationResult
from ..locations.coordinate import Coordinate
from ..registry import LOCALIZERS


def _check_skada():
    """Check if SKADA is installed."""
    try:
        import skada
        return True
    except ImportError:
        raise ImportError(
            "SKADA is required for transfer learning. "
            "Install with: pip install skada"
        )


@LOCALIZERS.register_module()
class TransferLocalizer(TraditionalLocalizer):
    """
    Transfer Learning Localizer using domain adaptation.

    Wraps SKADA methods to adapt models from source domain (labeled)
    to target domain (unlabeled). Useful for:
    - Cross-device adaptation (different phones)
    - Cross-time adaptation (signal drift over time)
    - Cross-environment adaptation (different buildings/floors)

    Args:
        method: Domain adaptation method
            - 'coral': CORAL (Correlation Alignment) - fast, stable
            - 'tca': TCA (Transfer Component Analysis) - classic
            - 'sa': Subspace Alignment
            - 'kmm': Kernel Mean Matching (instance reweighting)
            - 'jdot': Joint Distribution Optimal Transport
            - 'otda': Optimal Transport Domain Adaptation
        base_estimator: Base sklearn estimator for regression
            Default: KNeighborsRegressor(n_neighbors=5)
        n_components: Number of components for subspace methods (TCA, SA)
        kernel: Kernel for KMM ('rbf', 'linear', 'poly')
        **kwargs: Additional arguments for the DA method

    Example:
        >>> import indoorloc as iloc
        >>>
        >>> # Load source (labeled) and target (unlabeled) data
        >>> source_train, source_test = iloc.load_dataset("ujindoorloc")
        >>> target_data = iloc.load_dataset("target_building")
        >>>
        >>> # Create transfer localizer
        >>> model = iloc.TransferLocalizer(method='coral')
        >>>
        >>> # Adapt and train
        >>> model.fit(source_train, target_data=target_data)
        >>>
        >>> # Predict on target domain
        >>> result = model.predict(target_signal)
    """

    # Supported methods
    METHODS = {
        'coral': 'CORAL',
        'tca': 'TransferComponentAnalysis',
        'sa': 'SubspaceAlignment',
        'kmm': 'KMMReweight',
        'jdot': 'JDOTRegressor',
        'otda': 'OTMapping',
    }

    def __init__(
        self,
        method: str = 'coral',
        base_estimator: Any = None,
        n_components: Optional[int] = None,
        kernel: str = 'rbf',
        predict_floor: bool = True,
        predict_building: bool = True,
        **kwargs
    ):
        super().__init__(
            predict_floor=predict_floor,
            predict_building=predict_building,
            **kwargs
        )

        if method.lower() not in self.METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported: {list(self.METHODS.keys())}"
            )

        self.method = method.lower()
        self.n_components = n_components
        self.kernel = kernel
        self._da_kwargs = kwargs

        # Base estimator for coordinate regression
        if base_estimator is None:
            from sklearn.neighbors import KNeighborsRegressor
            base_estimator = KNeighborsRegressor(n_neighbors=5)
        self._base_estimator = base_estimator

        # DA models (created during fit)
        self._da_coord_model = None
        self._da_floor_model = None
        self._da_building_model = None

        # Store target domain features for prediction
        self._target_features = None

    @property
    def localizer_type(self) -> str:
        return f'transfer_{self.method}'

    def _create_da_model(self, base_estimator, is_classifier: bool = False):
        """Create domain adaptation model based on method."""
        _check_skada()

        if self.method == 'coral':
            from skada import CORAL
            return CORAL(base_estimator=base_estimator)

        elif self.method == 'tca':
            from skada import TransferComponentAnalysis
            n_comp = self.n_components or 50
            return TransferComponentAnalysis(
                base_estimator=base_estimator,
                n_components=n_comp
            )

        elif self.method == 'sa':
            from skada import SubspaceAlignment
            n_comp = self.n_components or 50
            return SubspaceAlignment(
                base_estimator=base_estimator,
                n_components=n_comp
            )

        elif self.method == 'kmm':
            from skada import KMMReweight
            # KMM requires sample_weight support
            if hasattr(base_estimator, 'set_fit_request'):
                base_estimator = base_estimator.set_fit_request(sample_weight=True)
            return KMMReweight(
                base_estimator=base_estimator,
                kernel=self.kernel
            )

        elif self.method == 'jdot':
            from skada import JDOTRegressor
            return JDOTRegressor(base_estimator=base_estimator)

        elif self.method == 'otda':
            from skada import OTMapping
            return OTMapping(base_estimator=base_estimator)

        else:
            raise ValueError(f"Method {self.method} not implemented")

    def _fit_impl(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        target_data: Optional[Union[List[BaseSignal], 'BaseDataset']] = None,
        **kwargs
    ) -> 'TransferLocalizer':
        """
        Train with domain adaptation.

        Args:
            signals: Source domain signals (labeled)
            locations: Source domain locations
            target_data: Target domain data (unlabeled signals or dataset)

        Returns:
            Self for method chaining
        """
        _check_skada()

        # Extract source features and labels
        Xs = np.array([self._extract_features(s) for s in signals])
        labels = self._extract_labels(locations)

        # Extract target features
        if target_data is not None:
            from ..datasets.base import BaseDataset
            if isinstance(target_data, BaseDataset):
                target_signals = target_data.signals
            else:
                target_signals = target_data
            Xt = np.array([self._extract_features(s) for s in target_signals])
        else:
            # No target data - use source as target (no adaptation)
            Xt = Xs

        self._target_features = Xt

        # Create sample_domain: positive for source, negative for target
        n_source = len(Xs)
        n_target = len(Xt)
        X_combined = np.vstack([Xs, Xt])
        sample_domain = np.concatenate([
            np.ones(n_source, dtype=int),      # source: positive
            -np.ones(n_target, dtype=int)      # target: negative
        ])

        # Labels: source has labels, target has NaN/-1
        y_coords = np.vstack([
            labels['coords'],
            np.full((n_target, 2), np.nan)
        ])

        # Fit coordinate model
        from sklearn.base import clone
        base_coord = clone(self._base_estimator)
        self._da_coord_model = self._create_da_model(base_coord)
        self._da_coord_model.fit(X_combined, y_coords, sample_domain=sample_domain)

        # Fit floor model if enabled
        if self.predict_floor:
            valid_floor = labels['floor'] >= 0
            if np.any(valid_floor) and np.unique(labels['floor'][valid_floor]).size > 1:
                from sklearn.neighbors import KNeighborsClassifier
                base_floor = KNeighborsClassifier(n_neighbors=5)
                self._da_floor_model = self._create_da_model(base_floor, is_classifier=True)

                y_floor = np.concatenate([
                    labels['floor'],
                    np.full(n_target, -1, dtype=int)
                ])
                try:
                    self._da_floor_model.fit(X_combined, y_floor, sample_domain=sample_domain)
                except Exception:
                    self._da_floor_model = None

        # Fit building model if enabled
        if self.predict_building:
            valid_building = labels['building'] >= 0
            if np.any(valid_building) and np.unique(labels['building'][valid_building]).size > 1:
                from sklearn.neighbors import KNeighborsClassifier
                base_building = KNeighborsClassifier(n_neighbors=5)
                self._da_building_model = self._create_da_model(base_building, is_classifier=True)

                y_building = np.concatenate([
                    labels['building'],
                    np.full(n_target, -1, dtype=int)
                ])
                try:
                    self._da_building_model.fit(X_combined, y_building, sample_domain=sample_domain)
                except Exception:
                    self._da_building_model = None

        self._is_trained = True
        return self

    def predict(self, signal: BaseSignal) -> LocalizationResult:
        """Predict location for a signal."""
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = self._extract_features(signal).reshape(1, -1)

        # Predict coordinates
        coords = self._da_coord_model.predict(X)[0]

        # Predict floor
        floor = None
        floor_confidence = 0.0
        if self._da_floor_model is not None:
            try:
                floor = int(self._da_floor_model.predict(X)[0])
                if hasattr(self._da_floor_model, 'predict_proba'):
                    floor_proba = self._da_floor_model.predict_proba(X)[0]
                    floor_confidence = float(np.max(floor_proba))
            except Exception:
                pass

        # Predict building
        building = None
        if self._da_building_model is not None:
            try:
                building = str(int(self._da_building_model.predict(X)[0]))
            except Exception:
                pass

        location = Location(
            coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
            floor=floor,
            building_id=building,
            confidence=0.8,  # Default confidence for DA
            floor_confidence=floor_confidence
        )

        return LocalizationResult(location=location)

    def predict_batch(self, signals: List[BaseSignal]) -> List[LocalizationResult]:
        """Predict locations for multiple signals."""
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = np.array([self._extract_features(s) for s in signals])
        coords_batch = self._da_coord_model.predict(X)

        floors_batch = None
        floor_confidences = None
        if self._da_floor_model is not None:
            try:
                floors_batch = self._da_floor_model.predict(X)
                if hasattr(self._da_floor_model, 'predict_proba'):
                    floor_proba = self._da_floor_model.predict_proba(X)
                    floor_confidences = np.max(floor_proba, axis=1)
            except Exception:
                pass

        buildings_batch = None
        if self._da_building_model is not None:
            try:
                buildings_batch = self._da_building_model.predict(X)
            except Exception:
                pass

        results = []
        for i in range(len(signals)):
            location = Location(
                coordinate=Coordinate(
                    x=float(coords_batch[i, 0]),
                    y=float(coords_batch[i, 1])
                ),
                floor=int(floors_batch[i]) if floors_batch is not None else None,
                building_id=str(int(buildings_batch[i])) if buildings_batch is not None else None,
                confidence=0.8,
                floor_confidence=float(floor_confidences[i]) if floor_confidences is not None else 0.0
            )
            results.append(LocalizationResult(location=location))

        return results

    def _get_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        return {
            'method': self.method,
            'n_components': self.n_components,
            'kernel': self.kernel,
            'da_coord_model': self._da_coord_model,
            'da_floor_model': self._da_floor_model,
            'da_building_model': self._da_building_model,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loading."""
        self.method = state['method']
        self.n_components = state['n_components']
        self.kernel = state['kernel']
        self._da_coord_model = state['da_coord_model']
        self._da_floor_model = state['da_floor_model']
        self._da_building_model = state['da_building_model']


# Convenience aliases
@LOCALIZERS.register_module()
class CORALLocalizer(TransferLocalizer):
    """CORAL (Correlation Alignment) Localizer."""

    def __init__(self, **kwargs):
        super().__init__(method='coral', **kwargs)

    @property
    def localizer_type(self) -> str:
        return 'coral'


@LOCALIZERS.register_module()
class TCALocalizer(TransferLocalizer):
    """TCA (Transfer Component Analysis) Localizer."""

    def __init__(self, n_components: int = 50, **kwargs):
        super().__init__(method='tca', n_components=n_components, **kwargs)

    @property
    def localizer_type(self) -> str:
        return 'tca'


@LOCALIZERS.register_module()
class KMMLocalizer(TransferLocalizer):
    """KMM (Kernel Mean Matching) Localizer."""

    def __init__(self, kernel: str = 'rbf', **kwargs):
        super().__init__(method='kmm', kernel=kernel, **kwargs)

    @property
    def localizer_type(self) -> str:
        return 'kmm'


__all__ = [
    'TransferLocalizer',
    'CORALLocalizer',
    'TCALocalizer',
    'KMMLocalizer',
]
