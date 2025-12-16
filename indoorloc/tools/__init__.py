"""CLI Entry Points for IndoorLoc.

This module provides command-line tools for training, testing, and benchmarking
indoor localization models.

Entry points (after pip install):
    - indoorloc-train: Train a model
    - indoorloc-test: Test a trained model
    - indoorloc-benchmark: Compare results against published benchmarks
"""
from .train import main as train_main
from .test import main as test_main
from .benchmark import main as benchmark_main

__all__ = ['train_main', 'test_main', 'benchmark_main']
