"""
k-Nearest Neighbors Localizer

Classic WiFi fingerprint-based localization using k-NN.
"""
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from ..base import TraditionalLocalizer
from ...signals.base import BaseSignal
from ...locations.location import Location, LocalizationResult
from ...locations.coordinate import Coordinate
from ...registry import LOCALIZERS


@LOCALIZERS.register_module()
class KNNLocalizer(TraditionalLocalizer):
    """
    k-Nearest Neighbors Localizer.

    Uses k-NN for:
    - Coordinate regression (position prediction)
    - Floor classification (optional)
    - Building classification (optional)

    Attributes:
        k: Number of neighbors
        weights: Weight function ('uniform' or 'distance')
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        algorithm: Algorithm for finding neighbors
    """

    def __init__(
        self,
        k: int = 5,
        weights: str = 'distance',
        metric: str = 'euclidean',
        algorithm: str = 'auto',
        predict_floor: bool = True,
        predict_building: bool = True,
        **kwargs
    ):
        """
        Initialize k-NN localizer.

        Args:
            k: Number of neighbors
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            algorithm: Algorithm ('auto', 'ball_tree', 'kd_tree', 'brute')
            predict_floor: Whether to predict floor
            predict_building: Whether to predict building
        """
        super().__init__(
            predict_floor=predict_floor,
            predict_building=predict_building,
            **kwargs
        )

        self.k = k
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm

        # Initialize models
        self._coord_model = KNeighborsRegressor(
            n_neighbors=k,
            weights=weights,
            metric=metric,
            algorithm=algorithm
        )

        if predict_floor:
            self._floor_model = KNeighborsClassifier(
                n_neighbors=k,
                weights=weights,
                metric=metric,
                algorithm=algorithm
            )

        if predict_building:
            self._building_model = KNeighborsClassifier(
                n_neighbors=k,
                weights=weights,
                metric=metric,
                algorithm=algorithm
            )

    @property
    def localizer_type(self) -> str:
        return 'knn'

    def _fit_impl(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        **kwargs
    ) -> 'KNNLocalizer':
        """
        Train the k-NN localizer.

        Args:
            signals: Training signals
            locations: Ground truth locations

        Returns:
            Self for method chaining
        """
        # Extract features
        X = np.array([self._extract_features(s) for s in signals])

        # Extract labels
        labels = self._extract_labels(locations)

        # Fit coordinate model
        self._coord_model.fit(X, labels['coords'])

        # Fit floor model if enabled
        if self.predict_floor and self._floor_model is not None:
            valid_floor = labels['floor'] >= 0
            if np.any(valid_floor):
                self._floor_model.fit(X[valid_floor], labels['floor'][valid_floor])

        # Fit building model if enabled
        if self.predict_building and self._building_model is not None:
            valid_building = labels['building'] >= 0
            if np.any(valid_building):
                self._building_model.fit(X[valid_building], labels['building'][valid_building])

        self._is_trained = True
        return self

    def predict(self, signal: BaseSignal) -> LocalizationResult:
        """
        Predict location for a signal.

        Args:
            signal: Input signal

        Returns:
            Localization result
        """
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = self._extract_features(signal).reshape(1, -1)

        # Predict coordinates
        coords = self._coord_model.predict(X)[0]

        # Predict floor
        floor = None
        floor_confidence = 0.0
        if self._floor_model is not None:
            try:
                floor = int(self._floor_model.predict(X)[0])
                floor_proba = self._floor_model.predict_proba(X)[0]
                floor_confidence = float(np.max(floor_proba))
            except Exception:
                pass

        # Predict building
        building = None
        if self._building_model is not None:
            try:
                building = str(int(self._building_model.predict(X)[0]))
            except Exception:
                pass

        # Calculate uncertainty based on k neighbors
        distances, _ = self._coord_model.kneighbors(X)
        uncertainty = float(np.mean(distances))

        # Create location
        location = Location(
            coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
            floor=floor,
            building_id=building,
            confidence=1.0 - min(uncertainty / 20.0, 1.0),  # Heuristic
            position_uncertainty=uncertainty,
            floor_confidence=floor_confidence
        )

        return LocalizationResult(location=location)

    def predict_batch(
        self,
        signals: List[BaseSignal]
    ) -> List[LocalizationResult]:
        """
        Predict locations for multiple signals efficiently.

        Args:
            signals: List of input signals

        Returns:
            List of localization results
        """
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = np.array([self._extract_features(s) for s in signals])

        # Batch predict coordinates
        coords_batch = self._coord_model.predict(X)

        # Batch predict floor
        floors_batch = None
        floor_confidences = None
        if self._floor_model is not None:
            try:
                floors_batch = self._floor_model.predict(X)
                floor_proba = self._floor_model.predict_proba(X)
                floor_confidences = np.max(floor_proba, axis=1)
            except Exception:
                pass

        # Batch predict building
        buildings_batch = None
        if self._building_model is not None:
            try:
                buildings_batch = self._building_model.predict(X)
            except Exception:
                pass

        # Batch get uncertainties
        distances, _ = self._coord_model.kneighbors(X)
        uncertainties = np.mean(distances, axis=1)

        # Create results
        results = []
        for i in range(len(signals)):
            location = Location(
                coordinate=Coordinate(
                    x=float(coords_batch[i, 0]),
                    y=float(coords_batch[i, 1])
                ),
                floor=int(floors_batch[i]) if floors_batch is not None else None,
                building_id=str(int(buildings_batch[i])) if buildings_batch is not None else None,
                confidence=1.0 - min(uncertainties[i] / 20.0, 1.0),
                position_uncertainty=float(uncertainties[i]),
                floor_confidence=float(floor_confidences[i]) if floor_confidences is not None else 0.0
            )
            results.append(LocalizationResult(location=location))

        return results

    def _get_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        return {
            'coord_model': self._coord_model,
            'floor_model': self._floor_model,
            'building_model': self._building_model,
            'k': self.k,
            'weights': self.weights,
            'metric': self.metric,
            'algorithm': self.algorithm,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loading."""
        self._coord_model = state['coord_model']
        self._floor_model = state['floor_model']
        self._building_model = state['building_model']
        self.k = state['k']
        self.weights = state['weights']
        self.metric = state['metric']
        self.algorithm = state['algorithm']


@LOCALIZERS.register_module()
class WKNNLocalizer(KNNLocalizer):
    """
    Weighted k-Nearest Neighbors Localizer.

    Same as KNNLocalizer but with distance weighting by default.
    """

    def __init__(self, k: int = 5, **kwargs):
        super().__init__(k=k, weights='distance', **kwargs)

    @property
    def localizer_type(self) -> str:
        return 'wknn'


__all__ = ['KNNLocalizer', 'WKNNLocalizer']
