"""
Benchmark Data for Indoor Localization Datasets

This module contains published benchmark results from academic papers
for various indoor localization datasets.

Data is manually curated from peer-reviewed publications.
"""
from ..benchmarks import DatasetBenchmarks

# Import all dataset benchmarks
from .ujindoorloc import BENCHMARKS as UJINDOORLOC_BENCHMARKS
from .tampere import BENCHMARKS as TAMPERE_BENCHMARKS
from .sodindoorloc import BENCHMARKS as SODINDOORLOC_BENCHMARKS

# Registry of all available benchmarks
BENCHMARKS_REGISTRY = {
    'ujindoorloc': UJINDOORLOC_BENCHMARKS,
    'uji': UJINDOORLOC_BENCHMARKS,  # Alias
    'tampere': TAMPERE_BENCHMARKS,
    'sodindoorloc': SODINDOORLOC_BENCHMARKS,
    'sod': SODINDOORLOC_BENCHMARKS,  # Alias
}

__all__ = [
    'BENCHMARKS_REGISTRY',
    'UJINDOORLOC_BENCHMARKS',
    'TAMPERE_BENCHMARKS',
    'SODINDOORLOC_BENCHMARKS',
]
