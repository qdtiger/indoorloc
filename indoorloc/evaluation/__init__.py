"""
Evaluation Module

Metrics and evaluation tools for indoor localization.
"""
from .metrics import (
    BaseMetric,
    MeanPositionError,
    MedianPositionError,
    PercentileError,
    MaxPositionError,
    RMSPositionError,
    FloorAccuracy,
    BuildingAccuracy,
    FloorBuildingAccuracy,
    CDFAnalysis,
    Evaluator,
    EvaluationResults,
)

from .benchmarks import (
    BenchmarkEntry,
    DatasetBenchmarks,
    ComparisonReport,
    register_benchmarks,
    get_benchmarks_for_dataset,
    list_datasets_with_benchmarks,
)

__all__ = [
    # Metrics
    'BaseMetric',
    'MeanPositionError',
    'MedianPositionError',
    'PercentileError',
    'MaxPositionError',
    'RMSPositionError',
    'FloorAccuracy',
    'BuildingAccuracy',
    'FloorBuildingAccuracy',
    'CDFAnalysis',
    'Evaluator',
    'EvaluationResults',
    # Benchmarks
    'BenchmarkEntry',
    'DatasetBenchmarks',
    'ComparisonReport',
    'register_benchmarks',
    'get_benchmarks_for_dataset',
    'list_datasets_with_benchmarks',
]
