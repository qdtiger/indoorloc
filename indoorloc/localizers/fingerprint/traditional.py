"""
Traditional ML Localizers (SVM, Random Forest)
"""
from typing import List, Optional, Dict, Any
import warnings

import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

from ..base import TraditionalLocalizer
from ...signals.base import BaseSignal
from ...locations.location import Location, LocalizationResult
from ...locations.coordinate import Coordinate
from ...registry import LOCALIZERS


@LOCALIZERS.register_module()
class SVMLocalizer(TraditionalLocalizer):
    """
    Support Vector Machine Localizer.

    Uses SVR for coordinate regression, SVC for floor/building classification.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 100.0,
        gamma: Any = 'scale',
        epsilon: float = 0.1,
        n_jobs: Optional[int] = None,
        predict_floor: bool = True,
        predict_building: bool = True,
        **kwargs
    ):
        super().__init__(
            predict_floor=predict_floor,
            predict_building=predict_building,
            **kwargs
        )

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_jobs = n_jobs

        base_svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self._coord_model = MultiOutputRegressor(base_svr, n_jobs=n_jobs)

        if predict_floor:
            self._floor_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        if predict_building:
            self._building_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    @property
    def localizer_type(self) -> str:
        return 'svm'

    def _fit_impl(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        **kwargs
    ) -> 'SVMLocalizer':
        X = np.array([self._extract_features(s) for s in signals])
        labels = self._extract_labels(locations)

        if len(signals) > 5000:
            warnings.warn(
                f"SVMLocalizer: Training on {len(signals)} samples may be slow. "
                "Consider using RandomForestLocalizer for large datasets.",
                UserWarning
            )

        self._coord_model.fit(X, labels['coords'])

        if self.predict_floor and self._floor_model is not None:
            valid = labels['floor'] >= 0
            if np.any(valid) and np.unique(labels['floor'][valid]).size > 1:
                self._floor_model.fit(X[valid], labels['floor'][valid])
            else:
                self._floor_model = None

        if self.predict_building and self._building_model is not None:
            valid = labels['building'] >= 0
            if np.any(valid) and np.unique(labels['building'][valid]).size > 1:
                self._building_model.fit(X[valid], labels['building'][valid])
            else:
                self._building_model = None

        self._is_trained = True
        return self

    def predict(self, signal: BaseSignal) -> LocalizationResult:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = self._extract_features(signal).reshape(1, -1)
        coords = self._coord_model.predict(X)[0]

        floor, floor_confidence = None, 0.0
        if self._floor_model is not None:
            try:
                floor = int(self._floor_model.predict(X)[0])
                floor_confidence = float(np.max(self._floor_model.predict_proba(X)[0]))
            except Exception:
                pass

        building = None
        if self._building_model is not None:
            try:
                building = str(int(self._building_model.predict(X)[0]))
            except Exception:
                pass

        location = Location(
            coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
            floor=floor,
            building_id=building,
            confidence=0.8,
            position_uncertainty=self.epsilon,
            floor_confidence=floor_confidence
        )
        return LocalizationResult(location=location)

    def predict_batch(self, signals: List[BaseSignal]) -> List[LocalizationResult]:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = np.array([self._extract_features(s) for s in signals])
        coords_batch = self._coord_model.predict(X)

        floors_batch, floor_confidences = None, None
        if self._floor_model is not None:
            try:
                floors_batch = self._floor_model.predict(X)
                floor_confidences = np.max(self._floor_model.predict_proba(X), axis=1)
            except Exception:
                pass

        buildings_batch = None
        if self._building_model is not None:
            try:
                buildings_batch = self._building_model.predict(X)
            except Exception:
                pass

        results = []
        for i in range(len(signals)):
            location = Location(
                coordinate=Coordinate(x=float(coords_batch[i, 0]), y=float(coords_batch[i, 1])),
                floor=int(floors_batch[i]) if floors_batch is not None else None,
                building_id=str(int(buildings_batch[i])) if buildings_batch is not None else None,
                confidence=0.8,
                position_uncertainty=self.epsilon,
                floor_confidence=float(floor_confidences[i]) if floor_confidences is not None else 0.0
            )
            results.append(LocalizationResult(location=location))
        return results

    def _get_state(self) -> Dict[str, Any]:
        return {
            'coord_model': self._coord_model,
            'floor_model': self._floor_model,
            'building_model': self._building_model,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'n_jobs': self.n_jobs,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self._coord_model = state['coord_model']
        self._floor_model = state['floor_model']
        self._building_model = state['building_model']
        self.kernel = state.get('kernel', 'rbf')
        self.C = state.get('C', 100.0)
        self.gamma = state.get('gamma', 'scale')
        self.epsilon = state.get('epsilon', 0.1)
        self.n_jobs = state.get('n_jobs', None)


@LOCALIZERS.register_module()
class RandomForestLocalizer(TraditionalLocalizer):
    """
    Random Forest Localizer.

    Uses RandomForestRegressor for coordinates, RandomForestClassifier for floor/building.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = 'sqrt',
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        predict_floor: bool = True,
        predict_building: bool = True,
        **kwargs
    ):
        super().__init__(
            predict_floor=predict_floor,
            predict_building=predict_building,
            **kwargs
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        rf_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self._coord_model = RandomForestRegressor(**rf_params)
        if predict_floor:
            self._floor_model = RandomForestClassifier(**rf_params)
        if predict_building:
            self._building_model = RandomForestClassifier(**rf_params)

    @property
    def localizer_type(self) -> str:
        return 'rf'

    def _fit_impl(
        self,
        signals: List[BaseSignal],
        locations: List[Location],
        **kwargs
    ) -> 'RandomForestLocalizer':
        X = np.array([self._extract_features(s) for s in signals])
        labels = self._extract_labels(locations)

        self._coord_model.fit(X, labels['coords'])

        if self.predict_floor and self._floor_model is not None:
            valid = labels['floor'] >= 0
            if np.any(valid) and np.unique(labels['floor'][valid]).size > 1:
                self._floor_model.fit(X[valid], labels['floor'][valid])
            else:
                self._floor_model = None

        if self.predict_building and self._building_model is not None:
            valid = labels['building'] >= 0
            if np.any(valid) and np.unique(labels['building'][valid]).size > 1:
                self._building_model.fit(X[valid], labels['building'][valid])
            else:
                self._building_model = None

        self._is_trained = True
        return self

    def predict(self, signal: BaseSignal) -> LocalizationResult:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = self._extract_features(signal).reshape(1, -1)
        coords = self._coord_model.predict(X)[0]

        floor, floor_confidence = None, 0.0
        if self._floor_model is not None:
            try:
                floor = int(self._floor_model.predict(X)[0])
                floor_confidence = float(np.max(self._floor_model.predict_proba(X)[0]))
            except Exception:
                pass

        building = None
        if self._building_model is not None:
            try:
                building = str(int(self._building_model.predict(X)[0]))
            except Exception:
                pass

        location = Location(
            coordinate=Coordinate(x=float(coords[0]), y=float(coords[1])),
            floor=floor,
            building_id=building,
            confidence=0.9,
            position_uncertainty=0.0,
            floor_confidence=floor_confidence
        )
        return LocalizationResult(location=location)

    def predict_batch(self, signals: List[BaseSignal]) -> List[LocalizationResult]:
        if not self._is_trained:
            raise RuntimeError("Localizer must be trained before prediction")

        X = np.array([self._extract_features(s) for s in signals])
        coords_batch = self._coord_model.predict(X)

        floors_batch, floor_confidences = None, None
        if self._floor_model is not None:
            try:
                floors_batch = self._floor_model.predict(X)
                floor_confidences = np.max(self._floor_model.predict_proba(X), axis=1)
            except Exception:
                pass

        buildings_batch = None
        if self._building_model is not None:
            try:
                buildings_batch = self._building_model.predict(X)
            except Exception:
                pass

        results = []
        for i in range(len(signals)):
            location = Location(
                coordinate=Coordinate(x=float(coords_batch[i, 0]), y=float(coords_batch[i, 1])),
                floor=int(floors_batch[i]) if floors_batch is not None else None,
                building_id=str(int(buildings_batch[i])) if buildings_batch is not None else None,
                confidence=0.9,
                position_uncertainty=0.0,
                floor_confidence=float(floor_confidences[i]) if floor_confidences is not None else 0.0
            )
            results.append(LocalizationResult(location=location))
        return results

    def _get_state(self) -> Dict[str, Any]:
        return {
            'coord_model': self._coord_model,
            'floor_model': self._floor_model,
            'building_model': self._building_model,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self._coord_model = state['coord_model']
        self._floor_model = state['floor_model']
        self._building_model = state['building_model']
        self.n_estimators = state.get('n_estimators', 200)
        self.max_depth = state.get('max_depth', None)
        self.min_samples_split = state.get('min_samples_split', 2)
        self.min_samples_leaf = state.get('min_samples_leaf', 1)
        self.max_features = state.get('max_features', 'sqrt')
        self.random_state = state.get('random_state', 42)
        self.n_jobs = state.get('n_jobs', None)


__all__ = ['SVMLocalizer', 'RandomForestLocalizer']
