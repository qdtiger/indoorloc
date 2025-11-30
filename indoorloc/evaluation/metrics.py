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


# ============================================================================
# Evaluation Results
# ============================================================================

class EvaluationResults:
    """Encapsulates evaluation results with convenient access.

    Provides attribute access to common metrics and visualization methods.

    Example:
        >>> results = model.evaluate(test_dataset)
        >>> print(f"Mean Error: {results.mean_error:.2f}m")
        >>> print(results.summary())
        >>> results.plot_cdf()
        >>> results.compare_benchmarks()  # Compare against published results
    """

    def __init__(
        self,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location],
        errors: Optional[np.ndarray] = None,
        dataset_name: Optional[str] = None,
    ):
        self.predictions = predictions
        self.ground_truths = ground_truths
        self._dataset_name = dataset_name

        # Compute errors
        if errors is None:
            self._errors = np.array([
                _extract_location(pred).distance_to(gt)
                for pred, gt in zip(predictions, ground_truths)
            ])
        else:
            self._errors = errors

        # Cache computed metrics
        self._cache: Dict[str, float] = {}

    @classmethod
    def from_predictions(
        cls,
        predictions: List[Union[Location, LocalizationResult]],
        ground_truths: List[Location],
    ) -> 'EvaluationResults':
        """Create EvaluationResults from predictions and ground truths."""
        return cls(predictions, ground_truths)

    @property
    def errors(self) -> np.ndarray:
        """Get all position errors as numpy array."""
        return self._errors

    @property
    def mean_error(self) -> float:
        """Mean position error in meters."""
        return float(np.mean(self._errors))

    @property
    def median_error(self) -> float:
        """Median position error in meters."""
        return float(np.median(self._errors))

    @property
    def std_error(self) -> float:
        """Standard deviation of position errors."""
        return float(np.std(self._errors))

    @property
    def min_error(self) -> float:
        """Minimum position error."""
        return float(np.min(self._errors))

    @property
    def max_error(self) -> float:
        """Maximum position error."""
        return float(np.max(self._errors))

    @property
    def rms_error(self) -> float:
        """Root mean square position error."""
        return float(np.sqrt(np.mean(self._errors ** 2)))

    @property
    def p75_error(self) -> float:
        """75th percentile error."""
        return float(np.percentile(self._errors, 75))

    @property
    def p90_error(self) -> float:
        """90th percentile error."""
        return float(np.percentile(self._errors, 90))

    @property
    def p95_error(self) -> float:
        """95th percentile error."""
        return float(np.percentile(self._errors, 95))

    @property
    def floor_accuracy(self) -> float:
        """Floor classification accuracy (%)."""
        if 'floor_acc' not in self._cache:
            correct = sum(
                1 for pred, gt in zip(self.predictions, self.ground_truths)
                if _extract_location(pred).floor == gt.floor
            )
            self._cache['floor_acc'] = correct / len(self.predictions) * 100
        return self._cache['floor_acc']

    @property
    def building_accuracy(self) -> float:
        """Building classification accuracy (%)."""
        if 'building_acc' not in self._cache:
            correct = sum(
                1 for pred, gt in zip(self.predictions, self.ground_truths)
                if _extract_location(pred).building_id == gt.building_id
            )
            self._cache['building_acc'] = correct / len(self.predictions) * 100
        return self._cache['building_acc']

    def percentile(self, p: float) -> float:
        """Get error at specified percentile."""
        return float(np.percentile(self._errors, p))

    def within_threshold(self, threshold: float) -> float:
        """Get percentage of samples within error threshold."""
        return float(np.mean(self._errors <= threshold) * 100)

    def summary(self, verbose: bool = True) -> str:
        """Generate a summary string of evaluation results.

        Args:
            verbose: If True, include more detailed statistics.

        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Samples: {len(self.predictions)}",
            "",
            "Position Error:",
            f"  Mean:   {self.mean_error:.2f} m",
            f"  Median: {self.median_error:.2f} m",
            f"  Std:    {self.std_error:.2f} m",
            f"  P75:    {self.p75_error:.2f} m",
            f"  P95:    {self.p95_error:.2f} m",
        ]

        if verbose:
            lines.extend([
                f"  Min:    {self.min_error:.2f} m",
                f"  Max:    {self.max_error:.2f} m",
                f"  RMS:    {self.rms_error:.2f} m",
            ])

        lines.extend([
            "",
            "Classification:",
            f"  Floor Accuracy:    {self.floor_accuracy:.1f}%",
            f"  Building Accuracy: {self.building_accuracy:.1f}%",
            "",
            "CDF:",
            f"  Within 1m:  {self.within_threshold(1):.1f}%",
            f"  Within 3m:  {self.within_threshold(3):.1f}%",
            f"  Within 5m:  {self.within_threshold(5):.1f}%",
            f"  Within 10m: {self.within_threshold(10):.1f}%",
            "=" * 50,
        ])

        return "\n".join(lines)

    def plot_cdf(self, ax=None, label: str = None, **kwargs):
        """Plot the Cumulative Distribution Function of errors.

        Args:
            ax: Matplotlib axes. If None, creates a new figure.
            label: Label for the curve.
            **kwargs: Additional arguments passed to plot().

        Returns:
            Matplotlib axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        sorted_errors = np.sort(self._errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        ax.plot(sorted_errors, cdf * 100, label=label, **kwargs)
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('CDF (%)')
        ax.set_title('Cumulative Distribution Function of Position Errors')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)

        if label:
            ax.legend()

        return ax

    def plot_error_histogram(self, ax=None, bins: int = 50, **kwargs):
        """Plot histogram of position errors.

        Args:
            ax: Matplotlib axes. If None, creates a new figure.
            bins: Number of histogram bins.
            **kwargs: Additional arguments passed to hist().

        Returns:
            Matplotlib axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(self._errors, bins=bins, edgecolor='black', alpha=0.7, **kwargs)
        ax.axvline(self.mean_error, color='r', linestyle='--', label=f'Mean: {self.mean_error:.2f}m')
        ax.axvline(self.median_error, color='g', linestyle='--', label=f'Median: {self.median_error:.2f}m')
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Position Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def to_dict(self) -> Dict[str, float]:
        """Convert results to dictionary."""
        return {
            'mean_error': self.mean_error,
            'median_error': self.median_error,
            'std_error': self.std_error,
            'min_error': self.min_error,
            'max_error': self.max_error,
            'rms_error': self.rms_error,
            'p75_error': self.p75_error,
            'p90_error': self.p90_error,
            'p95_error': self.p95_error,
            'floor_accuracy': self.floor_accuracy,
            'building_accuracy': self.building_accuracy,
        }

    # ========================================================================
    # Benchmark Comparison
    # ========================================================================

    def compare_benchmarks(self, dataset_name: Optional[str] = None) -> 'ComparisonReport':
        """Compare results against published paper benchmarks.

        This method allows you to see how your results compare to
        state-of-the-art and baseline methods from academic publications.

        Args:
            dataset_name: Dataset name to look up benchmarks for.
                If None, uses the dataset name from evaluation context.

        Returns:
            ComparisonReport object that can be printed or analyzed.

        Raises:
            ValueError: If no dataset name is provided and none was set.
            ValueError: If no benchmarks are available for the dataset.

        Example:
            >>> results = model.evaluate(test_dataset)
            >>> results.compare_benchmarks()
            # Prints formatted comparison table

            >>> report = results.compare_benchmarks()
            >>> print(report.beats_count())  # How many methods you beat
            >>> print(report.gap_to_sota())  # Gap to SOTA in meters
        """
        from .benchmarks import get_benchmarks_for_dataset, ComparisonReport

        # Determine dataset name
        name = dataset_name or self._dataset_name
        if name is None:
            raise ValueError(
                "No dataset name specified. Either:\n"
                "  1. Pass dataset_name parameter: results.compare_benchmarks('ujindoorloc')\n"
                "  2. Evaluate with dataset reference: model.evaluate(test_dataset)"
            )

        # Get benchmarks
        benchmarks = get_benchmarks_for_dataset(name)
        if benchmarks is None:
            from .benchmarks import list_datasets_with_benchmarks
            available = list_datasets_with_benchmarks()
            raise ValueError(
                f"No benchmarks available for dataset '{name}'.\n"
                f"Available datasets with benchmarks: {available}"
            )

        # Create and print comparison report
        report = ComparisonReport(self, benchmarks)
        report.print_table()
        return report

    def get_benchmarks(self, dataset_name: Optional[str] = None) -> Optional[Dict]:
        """Get benchmark data as a dictionary.

        Args:
            dataset_name: Dataset name to look up benchmarks for.

        Returns:
            Dictionary with benchmark data, or None if not available.
        """
        from .benchmarks import get_benchmarks_for_dataset

        name = dataset_name or self._dataset_name
        if name is None:
            return None

        benchmarks = get_benchmarks_for_dataset(name)
        if benchmarks is None:
            return None

        sota = benchmarks.get_sota()
        return {
            'dataset': benchmarks.dataset_name,
            'sota': {
                'method': sota.method if sota else None,
                'mean_error': sota.mean_error if sota else None,
                'floor_accuracy': sota.floor_accuracy if sota else None,
            },
            'entries': [
                {
                    'method': e.method,
                    'mean_error': e.mean_error,
                    'floor_accuracy': e.floor_accuracy,
                    'source': e.source,
                }
                for e in benchmarks.entries
            ]
        }

    def __repr__(self) -> str:
        return f"EvaluationResults(mean={self.mean_error:.2f}m, median={self.median_error:.2f}m, floor_acc={self.floor_accuracy:.1f}%)"

    def __str__(self) -> str:
        return self.summary(verbose=False)
