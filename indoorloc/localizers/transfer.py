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

    def __init__(
        self,
        method: str = 'CORAL',
        base_estimator: Any = None,
        predict_floor: bool = True,
        predict_building: bool = True,
        **kwargs
    ):
        """
        Initialize transfer localizer.

        Args:
            method: Any SKADA method name, e.g.:
                - Feature: 'CORAL', 'SubspaceAlignment', 'TransferComponentAnalysis'
                - Reweight: 'KMMReweight', 'KLIEPReweight', 'GaussianReweight'
                - OT: 'OTMapping', 'EntropicOTMapping', 'LinearMonge'
                - Deep: 'DANN', 'CDAN', 'MDD', 'MCC', 'DeepCORAL'
            base_estimator: Base sklearn estimator (default: KNeighborsRegressor)
            **kwargs: Arguments passed to SKADA method
        """
        super().__init__(
            predict_floor=predict_floor,
            predict_building=predict_building,
        )

        self.method = method
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
        return f'transfer_{self.method.lower()}'

    def _create_da_model(self, base_estimator, is_classifier: bool = False):
        """Create domain adaptation model by dynamically importing from SKADA."""
        _check_skada()
        import skada

        # Try to get the method class from skada
        method_class = None

        # Check common module locations
        for module_name in ['', 'feature_based', 'instance_based', 'deep']:
            try:
                if module_name:
                    module = getattr(skada, module_name, None)
                    if module:
                        method_class = getattr(module, self.method, None)
                else:
                    method_class = getattr(skada, self.method, None)

                if method_class is not None:
                    break
            except AttributeError:
                continue

        if method_class is None:
            raise ValueError(
                f"Method '{self.method}' not found in SKADA. "
                f"Check available methods at: https://scikit-adaptation.github.io/"
            )

        # For reweighting methods, need sample_weight support
        if 'Reweight' in self.method or 'KMM' in self.method:
            if hasattr(base_estimator, 'set_fit_request'):
                base_estimator = base_estimator.set_fit_request(sample_weight=True)

        # Create the DA model with base estimator and any extra kwargs
        return method_class(base_estimator=base_estimator, **self._da_kwargs)

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
            'da_kwargs': self._da_kwargs,
            'da_coord_model': self._da_coord_model,
            'da_floor_model': self._da_floor_model,
            'da_building_model': self._da_building_model,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loading."""
        self.method = state['method']
        self._da_kwargs = state.get('da_kwargs', {})
        self._da_coord_model = state['da_coord_model']
        self._da_floor_model = state['da_floor_model']
        self._da_building_model = state['da_building_model']


__all__ = ['TransferLocalizer']
