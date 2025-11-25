"""
Evaluation Metrics for Indoor Localization

Provides various metrics for evaluating indoor localization performance.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ..locations.location import Location, LocalizationResult
from ..registry import METRICS


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    All metric implementations should inherit from this class.
    """

    def __init__(self, **kwargs):
        self._results: List[float] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        pass

    @property
    def unit(self) -> str:
        """Return the unit of this metric (e.g., 'm', '%')."""
        return ''

    @abstractmethod
    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        """Compute the metric value.

        Args:
            predictions: List of predicted locations.
            ground_truths: List of ground truth locations.

        Returns:
            Metric value.
        """
        pass

    def reset(self) -> None:
        """Reset the metric state."""
        self._results = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _extract_location(pred: Union[Location, LocalizationResult]) -> Location:
    """Extract Location from prediction (handles both Location and LocalizationResult)."""
    if isinstance(pred, LocalizationResult):
        return pred.location
    return pred


# ============================================================================
# Position Error Metrics
# ============================================================================

@METRICS.register_module()
class MeanPositionError(BaseMetric):
    """Mean Position Error (MPE) in meters.

    Computes the average Euclidean distance between predicted and true positions.
    """

    @property
    def name(self) -> str:
        return 'Mean Position Error'

    @property
    def unit(self) -> str:
        return 'm'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error)

        return float(np.mean(errors))


@METRICS.register_module()
class MedianPositionError(BaseMetric):
    """Median Position Error in meters.

    More robust to outliers than mean error.
    """

    @property
    def name(self) -> str:
        return 'Median Position Error'

    @property
    def unit(self) -> str:
        return 'm'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error)

        return float(np.median(errors))


@METRICS.register_module()
class PercentileError(BaseMetric):
    """Percentile Position Error in meters.

    Commonly used percentiles: 75th, 90th, 95th.

    Args:
        percentile: The percentile to compute (0-100).
    """

    def __init__(self, percentile: float = 75, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile

    @property
    def name(self) -> str:
        return f'{self.percentile}th Percentile Error'

    @property
    def unit(self) -> str:
        return 'm'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error)

        return float(np.percentile(errors, self.percentile))


@METRICS.register_module()
class MaxPositionError(BaseMetric):
    """Maximum Position Error in meters."""

    @property
    def name(self) -> str:
        return 'Max Position Error'

    @property
    def unit(self) -> str:
        return 'm'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error)

        return float(np.max(errors))


@METRICS.register_module()
class RMSPositionError(BaseMetric):
    """Root Mean Square Position Error in meters."""

    @property
    def name(self) -> str:
        return 'RMS Position Error'

    @property
    def unit(self) -> str:
        return 'm'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error ** 2)

        return float(np.sqrt(np.mean(errors)))


# ============================================================================
# Classification Metrics
# ============================================================================

@METRICS.register_module()
class FloorAccuracy(BaseMetric):
    """Floor Classification Accuracy.

    Percentage of samples where the floor is correctly predicted.
    """

    @property
    def name(self) -> str:
        return 'Floor Accuracy'

    @property
    def unit(self) -> str:
        return '%'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            if pred_loc.floor == gt.floor:
                correct += 1

        return float(correct / len(predictions) * 100)


@METRICS.register_module()
class BuildingAccuracy(BaseMetric):
    """Building Classification Accuracy.

    Percentage of samples where the building is correctly predicted.
    """

    @property
    def name(self) -> str:
        return 'Building Accuracy'

    @property
    def unit(self) -> str:
        return '%'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            if pred_loc.building_id == gt.building_id:
                correct += 1

        return float(correct / len(predictions) * 100)


@METRICS.register_module()
class FloorBuildingAccuracy(BaseMetric):
    """Combined Floor and Building Accuracy.

    Percentage of samples where both floor and building are correct.
    """

    @property
    def name(self) -> str:
        return 'Floor+Building Accuracy'

    @property
    def unit(self) -> str:
        return '%'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> float:
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            if pred_loc.floor == gt.floor and pred_loc.building_id == gt.building_id:
                correct += 1

        return float(correct / len(predictions) * 100)


# ============================================================================
# CDF Analysis
# ============================================================================

@METRICS.register_module()
class CDFAnalysis(BaseMetric):
    """Cumulative Distribution Function Analysis.

    Computes the percentage of samples within various error thresholds.

    Args:
        error_thresholds: List of error thresholds in meters.
    """

    def __init__(
        self,
        error_thresholds: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.error_thresholds = error_thresholds or [1, 2, 3, 5, 10, 15, 20]

    @property
    def name(self) -> str:
        return 'CDF Analysis'

    def compute(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> Dict[str, float]:
        """Compute CDF values.

        Returns:
            Dictionary mapping threshold to percentage of samples within that threshold.
        """
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error)

        errors = np.array(errors)
        results = {}

        for threshold in self.error_thresholds:
            percentage = float(np.mean(errors <= threshold) * 100)
            results[f'within_{threshold}m'] = percentage

        return results

    def get_cdf_curve(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location],
        num_points: int = 100
    ) -> tuple:
        """Get CDF curve data for plotting.

        Args:
            predictions: Predicted locations.
            ground_truths: Ground truth locations.
            num_points: Number of points on the curve.

        Returns:
            Tuple of (error_values, cdf_values).
        """
        errors = []
        for pred, gt in zip(predictions, ground_truths):
            pred_loc = _extract_location(pred)
            error = pred_loc.distance_to(gt)
            errors.append(error)

        errors = np.array(errors)
        sorted_errors = np.sort(errors)

        # Generate CDF
        error_values = np.linspace(0, sorted_errors[-1], num_points)
        cdf_values = np.array([np.mean(errors <= e) for e in error_values])

        return error_values, cdf_values


# ============================================================================
# Evaluator Class
# ============================================================================

class Evaluator:
    """Comprehensive evaluator for indoor localization.

    Combines multiple metrics and provides a unified evaluation interface.

    Args:
        metrics: List of metric instances or config dicts.

    Example:
        >>> evaluator = Evaluator([
        ...     MeanPositionError(),
        ...     MedianPositionError(),
        ...     FloorAccuracy(),
        ...     {'type': 'PercentileError', 'percentile': 95}
        ... ])
        >>> results = evaluator.evaluate(predictions, ground_truths)
    """

    def __init__(self, metrics: Optional[List[Union[BaseMetric, Dict]]] = None):
        self.metrics: List[BaseMetric] = []

        if metrics is None:
            # Default metrics
            metrics = [
                MeanPositionError(),
                MedianPositionError(),
                PercentileError(percentile=75),
                PercentileError(percentile=95),
                FloorAccuracy(),
                BuildingAccuracy(),
            ]

        for metric in metrics:
            if isinstance(metric, BaseMetric):
                self.metrics.append(metric)
            elif isinstance(metric, dict):
                # Build from config
                self.metrics.append(METRICS.build(metric))
            else:
                raise TypeError(f"Invalid metric type: {type(metric)}")

    def evaluate(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> Dict[str, Any]:
        """Evaluate predictions against ground truth.

        Args:
            predictions: List of predicted locations.
            ground_truths: List of ground truth locations.

        Returns:
            Dictionary of metric names to values.
        """
        results = {}

        for metric in self.metrics:
            value = metric.compute(predictions, ground_truths)

            if isinstance(value, dict):
                # CDF analysis returns a dict
                results[metric.name] = value
            else:
                # Format with unit
                key = f"{metric.name} ({metric.unit})" if metric.unit else metric.name
                results[key] = value

        return results

    def print_results(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location]
    ) -> None:
        """Evaluate and print results in a formatted table."""
        results = self.evaluate(predictions, ground_truths)

        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)

        for name, value in results.items():
            if isinstance(value, dict):
                print(f"\n{name}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.2f}%")
            else:
                print(f"{name}: {value:.4f}")

        print("=" * 50 + "\n")

    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics]
        return f"Evaluator(metrics={metric_names})"
