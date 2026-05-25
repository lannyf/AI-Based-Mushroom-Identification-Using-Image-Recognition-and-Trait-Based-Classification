"""Benchmark runners — one wrapper per identification method.

Each runner adapts a model from ``models/`` to the common
``BenchmarkRunner`` interface so the benchmark engine can treat
them uniformly.
"""

from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.llm_standalone_runner import LLMStandaloneRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.tree_runner import TreeRunner
from benchmarks.runners.unified_runner import UnifiedRunner

__all__ = [
    "BenchmarkRunner",
    "RunnerResult",
    "CNNRunner",
    "LLMStandaloneRunner",
    "TraitDBRunner",
    "TreeRunner",
    "UnifiedRunner",
]
