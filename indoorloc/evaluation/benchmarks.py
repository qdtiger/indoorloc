"""
Benchmark Comparison System

Allows users to compare their results against published paper benchmarks.

Example:
    >>> results = model.fit(train).evaluate(test)
    >>> results.compare_benchmarks()
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import EvaluationResults


@dataclass
class BenchmarkEntry:
    """A single benchmark result from a paper.

    Attributes:
        method: Method/model name (e.g., "CNNLoc", "WKNN")
        mean_error: Mean position error in meters
        median_error: Median position error in meters (optional)
        floor_accuracy: Floor classification accuracy as fraction 0-1 (optional)
        building_accuracy: Building classification accuracy as fraction 0-1 (optional)
        source: Paper citation or source
        year: Publication year
        is_sota: Whether this is the current state-of-the-art
        notes: Additional notes about the benchmark
    """
    method: str
    mean_error: float
    median_error: Optional[float] = None
    floor_accuracy: Optional[float] = None
    building_accuracy: Optional[float] = None
    source: str = ""
    year: Optional[int] = None
    is_sota: bool = False
    notes: str = ""


@dataclass
class DatasetBenchmarks:
    """Collection of benchmarks for a specific dataset.

    Attributes:
        dataset_name: Name of the dataset (e.g., "ujindoorloc")
        display_name: Human-readable name for display
        entries: List of benchmark entries
        default_metric: Primary metric for comparison ("mean_error", "median_error")
    """
    dataset_name: str
    display_name: str
    entries: List[BenchmarkEntry] = field(default_factory=list)
    default_metric: str = "mean_error"

    def get_sota(self) -> Optional[BenchmarkEntry]:
        """Get the SOTA (state-of-the-art) benchmark entry."""
        for entry in self.entries:
            if entry.is_sota:
                return entry
        # If no explicit SOTA, return entry with lowest mean error
        if self.entries:
            return min(self.entries, key=lambda e: e.mean_error)
        return None

    def sorted_by_error(self, ascending: bool = True) -> List[BenchmarkEntry]:
        """Return entries sorted by mean error."""
        return sorted(self.entries, key=lambda e: e.mean_error, reverse=not ascending)


class ComparisonReport:
    """Report comparing user results against benchmarks.

    Provides formatted output and analysis of how user results compare
    to published benchmarks.
    """

    def __init__(
        self,
        user_result: 'EvaluationResults',
        benchmarks: DatasetBenchmarks,
    ):
        self.user_result = user_result
        self.benchmarks = benchmarks
        self._sota = benchmarks.get_sota()

    @property
    def user_mean_error(self) -> float:
        """User's mean position error."""
        return self.user_result.mean_error

    @property
    def user_floor_accuracy(self) -> Optional[float]:
        """User's floor accuracy (as fraction 0-1)."""
        try:
            return self.user_result.floor_accuracy / 100  # Convert from % to fraction
        except:
            return None

    def beats_count(self) -> int:
        """Number of benchmarks the user result beats (lower error is better)."""
        return sum(1 for b in self.benchmarks.entries if self.user_mean_error < b.mean_error)

    def gap_to_sota(self) -> Optional[float]:
        """Gap to SOTA in meters (positive = worse than SOTA)."""
        if self._sota is None:
            return None
        return self.user_mean_error - self._sota.mean_error

    def gap_to_sota_percent(self) -> Optional[float]:
        """Gap to SOTA as percentage (100% = same as SOTA, >100% = worse)."""
        if self._sota is None or self._sota.mean_error == 0:
            return None
        return (self.user_mean_error / self._sota.mean_error) * 100

    def ranking(self) -> int:
        """User's ranking among all methods (1 = best)."""
        better_count = sum(1 for b in self.benchmarks.entries if b.mean_error < self.user_mean_error)
        return better_count + 1  # 1-indexed

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        print(self._build_table())

    def _build_table(self) -> str:
        """Build the formatted comparison table string."""
        lines = []

        # Header
        title = f"{self.benchmarks.display_name} Benchmark Comparison"
        width = 76
        lines.append("=" * width)
        lines.append(f"{title:^{width}}")
        lines.append("=" * width)

        # Table header
        lines.append(f"{'Method':<24} {'Mean Error':>12} {'Floor Acc':>12} {'Source':<24}")
        lines.append("-" * width)

        # User result (highlighted)
        floor_str = f"{self.user_floor_accuracy * 100:.1f}%" if self.user_floor_accuracy else "-"
        lines.append(f"{'>>> Your Result <<<':<24} {self.user_mean_error:>10.2f}m {floor_str:>12} {'-':<24}")
        lines.append("-" * width)

        # Benchmark entries (sorted by error)
        for entry in self.benchmarks.sorted_by_error():
            method_name = entry.method
            if entry.is_sota:
                method_name = f"[SOTA] {entry.method}"

            floor_str = f"{entry.floor_accuracy * 100:.1f}%" if entry.floor_accuracy else "-"

            # Truncate source if too long
            source = entry.source[:22] + ".." if len(entry.source) > 24 else entry.source
            if entry.year:
                source = f"{source} ({entry.year})" if len(source) < 18 else source

            lines.append(f"{method_name:<24} {entry.mean_error:>10.2f}m {floor_str:>12} {source:<24}")

        lines.append("=" * width)

        # Summary
        lines.append("")
        total = len(self.benchmarks.entries)
        beats = self.beats_count()

        if beats > 0:
            lines.append(f"  Your result beats {beats}/{total} published methods")
        else:
            lines.append(f"  Your result does not beat any of the {total} published methods")

        if self._sota:
            gap = self.gap_to_sota()
            gap_pct = self.gap_to_sota_percent()
            if gap <= 0:
                lines.append(f"  Congratulations! You achieved SOTA performance!")
            else:
                lines.append(f"  Gap to SOTA: {gap:.2f}m ({gap_pct:.0f}% of SOTA error)")

        rank = self.ranking()
        lines.append(f"  Your ranking: #{rank} out of {total + 1} methods")
        lines.append("")

        # Add references section (deduplicated by first author/short name)
        lines.append("-" * width)
        lines.append("References (for full citations, see indoorloc documentation):")
        seen_keys = set()
        ref_num = 1
        for entry in self.benchmarks.sorted_by_error():
            if not entry.source:
                continue
            # Create a dedup key from first part of source
            key = entry.source.split(',')[0].strip().lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # Format reference
            source_short = entry.source if len(entry.source) <= 70 else entry.source[:67] + "..."
            lines.append(f"  [{ref_num}] {source_short}")
            ref_num += 1
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert comparison to dictionary format."""
        return {
            'user_result': {
                'mean_error': self.user_mean_error,
                'floor_accuracy': self.user_floor_accuracy,
            },
            'beats_count': self.beats_count(),
            'total_benchmarks': len(self.benchmarks.entries),
            'gap_to_sota': self.gap_to_sota(),
            'gap_to_sota_percent': self.gap_to_sota_percent(),
            'ranking': self.ranking(),
            'sota': {
                'method': self._sota.method if self._sota else None,
                'mean_error': self._sota.mean_error if self._sota else None,
            },
            'benchmarks': [
                {
                    'method': b.method,
                    'mean_error': b.mean_error,
                    'floor_accuracy': b.floor_accuracy,
                    'source': b.source,
                    'is_sota': b.is_sota,
                }
                for b in self.benchmarks.entries
            ]
        }

    def __repr__(self) -> str:
        return f"ComparisonReport(user_error={self.user_mean_error:.2f}m, beats={self.beats_count()}/{len(self.benchmarks.entries)}, ranking=#{self.ranking()})"

    def __str__(self) -> str:
        return self._build_table()


# ============================================================================
# Benchmark Registry
# ============================================================================

# Global registry of dataset benchmarks
_BENCHMARK_REGISTRY: Dict[str, DatasetBenchmarks] = {}


def register_benchmarks(dataset_name: str, benchmarks: DatasetBenchmarks) -> None:
    """Register benchmarks for a dataset."""
    _BENCHMARK_REGISTRY[dataset_name.lower()] = benchmarks


def get_benchmarks_for_dataset(dataset_name: str) -> Optional[DatasetBenchmarks]:
    """Get benchmarks for a dataset by name.

    Args:
        dataset_name: Dataset name (case-insensitive)

    Returns:
        DatasetBenchmarks if found, None otherwise
    """
    if dataset_name is None:
        return None
    return _BENCHMARK_REGISTRY.get(dataset_name.lower())


def list_datasets_with_benchmarks() -> List[str]:
    """List all datasets that have benchmark data available."""
    return list(_BENCHMARK_REGISTRY.keys())


# ============================================================================
# Load benchmark data
# ============================================================================

def _load_benchmark_data():
    """Load all benchmark data from benchmark_data submodule."""
    try:
        from .benchmark_data import BENCHMARKS_REGISTRY
        for name, benchmarks in BENCHMARKS_REGISTRY.items():
            register_benchmarks(name, benchmarks)
    except ImportError:
        pass  # No benchmark data available yet


# Load benchmarks on module import
_load_benchmark_data()


__all__ = [
    'BenchmarkEntry',
    'DatasetBenchmarks',
    'ComparisonReport',
    'register_benchmarks',
    'get_benchmarks_for_dataset',
    'list_datasets_with_benchmarks',
]
