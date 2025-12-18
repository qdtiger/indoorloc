"""
Ensemble and Stacking Localizers

Fusion methods that combine multiple base localizers.
"""
from typing import List, Optional, Dict, Any, Union
from collections import Counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from .base import BaseLocalizer
from ..signals.base import BaseSignal
from ..locations.location import Location, LocalizationResult
from ..locations.coordinate import Coordinate
from ..registry import LOCALIZERS


@LOCALIZERS.register_module()
class EnsembleLocalizer(BaseLocalizer):
    """
    Ensemble Localizer combining multiple base localizers.

    Uses weighted averaging for coordinates and majority voting for floor/building.

    Example:
        >>> knn = iloc.KNNLocalizer(k=5)
        >>> rf = iloc.RandomForestLocalizer(n_estimators=100)
        >>> ensemble = iloc.EnsembleLocalizer(localizers=[knn, rf])
        >>> ensemble.fit(train_dataset)
        >>> result = ensemble.predict(signal)
    """

    def __init__(
        self,
        localizers: Optional[List[BaseLocalizer]] = None,
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize ensemble localizer.

        Args:
            localizers: List of base localizers to combine
            weights: Optional weights for each localizer (default: equal weights)
        """
        super().__init__(**kwargs)
        self.localizers = localizers or []
        self.weights = weights

    @property
    def localizer_type(self) -> str:
        return 'ensemble'

    def add_localizer(self, localizer: BaseLocalizer, weight: float = 1.0) -> 'EnsembleLocalizer':
        """Add a localizer to the ensemble."""
        self.localizers.append(localizer)
        if self.weights is None:
            self.weights = [1.0] * len(self.localizers)
        else:
            self.weights.append(weight)
        return self

    def _fit_impl(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        **kwargs
    ) -> 'EnsembleLocalizer':
        if not self.localizers:
            raise ValueError("No localizers in ensemble. Add localizers first.")

        for loc in self.localizers:
            loc.fit(signals, locations, **kwargs)

        if self.weights is None:
            self.weights = [1.0] * len(self.localizers)

        self._is_trained = True
        return self

    def predict(self, signal: BaseSignal) -> LocalizationResult:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        predictions = [loc.predict(signal) for loc in self.localizers]
        return self._combine_predictions(predictions)

    def predict_batch(self, signals: List[BaseSignal]) -> List[LocalizationResult]:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        all_preds = [loc.predict_batch(signals) for loc in self.localizers]
        results = []
        for i in range(len(signals)):
            preds_i = [all_preds[j][i] for j in range(len(self.localizers))]
            results.append(self._combine_predictions(preds_i))
        return results

    def _combine_predictions(self, predictions: List[LocalizationResult]) -> LocalizationResult:
        """Combine predictions via weighted average (coords) and voting (floor/building)."""
        weights = np.array(self.weights) / np.sum(self.weights)

        # Weighted average for coordinates
        coords = np.array([
            [p.location.coordinate.x, p.location.coordinate.y]
            for p in predictions
        ])
        avg_coords = np.average(coords, axis=0, weights=weights)

        # Majority voting for floor
        floors = [p.location.floor for p in predictions if p.location.floor is not None]
        floor = Counter(floors).most_common(1)[0][0] if floors else None

        # Majority voting for building
        buildings = [p.location.building_id for p in predictions if p.location.building_id is not None]
        building = Counter(buildings).most_common(1)[0][0] if buildings else None

        # Average confidence
        confidences = [p.location.confidence for p in predictions if p.location.confidence]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.8

        location = Location(
            coordinate=Coordinate(x=float(avg_coords[0]), y=float(avg_coords[1])),
            floor=floor,
            building_id=building,
            confidence=avg_confidence,
            position_uncertainty=0.0,
            floor_confidence=0.0
        )
        return LocalizationResult(location=location)

    def _get_state(self) -> Dict[str, Any]:
        return {
            'localizers': self.localizers,
            'weights': self.weights,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.localizers = state['localizers']
        self.weights = state['weights']


@LOCALIZERS.register_module()
class StackingLocalizer(BaseLocalizer):
    """
    Stacking Localizer using meta-learner on base localizer predictions.

    First-level: base localizers predict coordinates
    Second-level: meta-learner combines predictions

    Example:
        >>> knn = iloc.KNNLocalizer(k=5)
        >>> rf = iloc.RandomForestLocalizer(n_estimators=100)
        >>> stacking = iloc.StackingLocalizer(localizers=[knn, rf], meta_learner='ridge')
        >>> stacking.fit(train_dataset)
        >>> result = stacking.predict(signal)
    """

    def __init__(
        self,
        localizers: Optional[List[BaseLocalizer]] = None,
        meta_learner: str = 'ridge',
        use_original_features: bool = False,
        **kwargs
    ):
        """
        Initialize stacking localizer.

        Args:
            localizers: List of base localizers
            meta_learner: Type of meta-learner ('ridge' or 'rf')
            use_original_features: Whether to include original signal features
        """
        super().__init__(**kwargs)
        self.localizers = localizers or []
        self.meta_learner_type = meta_learner
        self.use_original_features = use_original_features
        self._meta_model = None

    @property
    def localizer_type(self) -> str:
        return 'stacking'

    def add_localizer(self, localizer: BaseLocalizer) -> 'StackingLocalizer':
        """Add a localizer to the stack."""
        self.localizers.append(localizer)
        return self

    def _fit_impl(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        **kwargs
    ) -> 'StackingLocalizer':
        if not self.localizers:
            raise ValueError("No localizers in stack. Add localizers first.")

        # Train base localizers
        for loc in self.localizers:
            loc.fit(signals, locations, **kwargs)

        # Generate meta-features from base predictions
        meta_features = self._generate_meta_features(signals)

        # Extract target coordinates
        y = np.array([[loc.coordinate.x, loc.coordinate.y] for loc in locations])

        # Train meta-learner
        if self.meta_learner_type == 'ridge':
            self._meta_model = Ridge(alpha=1.0)
        else:
            self._meta_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        self._meta_model.fit(meta_features, y)
        self._is_trained = True
        return self

    def _generate_meta_features(self, signals: List[BaseSignal]) -> np.ndarray:
        """Generate meta-features from base localizer predictions."""
        features_list = []

        for loc in self.localizers:
            preds = loc.predict_batch(signals)
            coords = np.array([[p.location.coordinate.x, p.location.coordinate.y] for p in preds])
            features_list.append(coords)

        meta_features = np.hstack(features_list)

        if self.use_original_features:
            original = np.array([s.to_numpy() for s in signals])
            meta_features = np.hstack([meta_features, original])

        return meta_features

    def predict(self, signal: BaseSignal) -> LocalizationResult:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        meta_features = self._generate_meta_features([signal])
        coords = self._meta_model.predict(meta_features)[0]

        # Get floor/building from base localizers via voting
        base_preds = [loc.predict(signal) for loc in self.localizers]

        floors = [p.location.floor for p in base_preds if p.location.floor is not None]
        floor = Counter(floors).most_common(1)[0][0] if floors else None

        buildings = [p.location.building_id for p in base_preds if p.location.building_id is not None]
        building = Counter(buildings).most_common(1)[0][0] if buildings else None

        location = Location(
            coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
            floor=floor,
            building_id=building,
            confidence=0.85,
            position_uncertainty=0.0,
            floor_confidence=0.0
        )
        return LocalizationResult(location=location)

    def predict_batch(self, signals: List[BaseSignal]) -> List[LocalizationResult]:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        meta_features = self._generate_meta_features(signals)
        coords_batch = self._meta_model.predict(meta_features)

        # Get floor/building from base localizers
        all_preds = [loc.predict_batch(signals) for loc in self.localizers]

        results = []
        for i in range(len(signals)):
            preds_i = [all_preds[j][i] for j in range(len(self.localizers))]

            floors = [p.location.floor for p in preds_i if p.location.floor is not None]
            floor = Counter(floors).most_common(1)[0][0] if floors else None

            buildings = [p.location.building_id for p in preds_i if p.location.building_id is not None]
            building = Counter(buildings).most_common(1)[0][0] if buildings else None

            location = Location(
                coordinate=Coordinate(x=float(coords_batch[i, 0]), y=float(coords_batch[i, 1])),
                floor=floor,
                building_id=building,
                confidence=0.85,
                position_uncertainty=0.0,
                floor_confidence=0.0
            )
            results.append(LocalizationResult(location=location))

        return results

    def _get_state(self) -> Dict[str, Any]:
        return {
            'localizers': self.localizers,
            'meta_learner_type': self.meta_learner_type,
            'meta_model': self._meta_model,
            'use_original_features': self.use_original_features,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.localizers = state['localizers']
        self.meta_learner_type = state['meta_learner_type']
        self._meta_model = state['meta_model']
        self.use_original_features = state.get('use_original_features', False)


__all__ = ['EnsembleLocalizer', 'StackingLocalizer']
