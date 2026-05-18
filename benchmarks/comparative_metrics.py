"""Metrics for the comparative benchmark.

Computes per-method accuracy, coverage, and agreement-state statistics
across CNN, Tree, DB, LLM (System A) and Unified (System B).
"""

from collections import Counter
from typing import Dict, List

from benchmarks.runners.base import RunnerResult


# ---------------------------------------------------------------------------
# Basic accuracy / coverage
# ---------------------------------------------------------------------------


def compute_accuracy(results: List[RunnerResult], ground_truth: List[str]) -> float:
    """Top-1 accuracy over covered predictions only."""
    correct = total = 0
    for r, gt in zip(results, ground_truth):
        if not r.coverage:
            continue
        total += 1
        if r.top_species == gt:
            correct += 1
    return correct / total if total > 0 else 0.0


def compute_coverage(results: List[RunnerResult]) -> float:
    return sum(1 for r in results if r.coverage) / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Agreement evaluator (cross-method)
# ---------------------------------------------------------------------------


def evaluate_agreement(
    cnn: RunnerResult,
    tree: RunnerResult,
    db: RunnerResult,
    unified: RunnerResult,
) -> str:
    """Classify agreement among the four prediction sources.

    Returns one of:
      * ``agree`` — all conclusive sources predict the same species.
      * ``partial`` — a majority (≥3 sources or 2/2) agree on one species.
      * ``disagree`` — no majority; sources split across multiple species.
      * ``inconclusive`` — fewer than 2 sources produced a prediction.
    """
    signals: Dict[str, str] = {}
    for key, res in (("cnn", cnn), ("tree", tree), ("db", db), ("unified", unified)):
        if res.coverage and res.top_species and res.top_species.lower() not in (
            "unknown",
            "error",
            "unable to parse",
            "none",
            "n/a",
        ):
            signals[key] = res.top_species.lower()

    valid = list(signals.values())
    if len(valid) < 2:
        return "inconclusive"

    unique = set(valid)
    if len(unique) == 1:
        return "agree"

    counts = Counter(valid)
    most_common = counts.most_common(1)[0]
    if most_common[1] >= 3 or (len(valid) == 2 and most_common[1] == 2):
        return "partial" if len(unique) > 1 else "agree"

    if most_common[1] == 1:
        return "disagree"

    return "partial"


# ---------------------------------------------------------------------------
# Standalone outperformance helpers
# ---------------------------------------------------------------------------


def unified_outperforms_all(
    unified: RunnerResult,
    cnn: RunnerResult,
    tree: RunnerResult,
    db: RunnerResult,
    llm: RunnerResult,
    ground_truth: str,
) -> bool:
    """True when Unified is correct and all standalone methods are wrong."""
    if not unified.coverage or unified.top_species != ground_truth:
        return False
    standalone_wrong = 0
    standalone_total = 0
    for res in (cnn, tree, db, llm):
        if res.coverage:
            standalone_total += 1
            if res.top_species != ground_truth:
                standalone_wrong += 1
    return standalone_total > 0 and standalone_wrong == standalone_total


def unified_wrong_but_standalone_right(
    unified: RunnerResult,
    cnn: RunnerResult,
    tree: RunnerResult,
    db: RunnerResult,
    llm: RunnerResult,
    ground_truth: str,
) -> bool:
    """True when Unified is wrong but at least one standalone method is correct."""
    if not unified.coverage or unified.top_species == ground_truth:
        return False
    for res in (cnn, tree, db, llm):
        if res.coverage and res.top_species == ground_truth:
            return True
    return False
