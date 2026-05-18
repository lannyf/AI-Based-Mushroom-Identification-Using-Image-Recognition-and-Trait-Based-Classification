#!/usr/bin/env python3
"""Comparative benchmark runner for the unified mushroom identification pipeline.

Evaluates CNN, decision-tree, trait-database, and unified-pipeline methods on the
same specimens and produces side-by-side comparison reports.

Usage::

    python -m benchmarks.run_comparative \
        --manifest benchmarks/evaluation_manifest.csv \
        --output-dir artifacts/benchmarks/comparative

The manifest is a CSV with columns:
  specimen_id, above_image_path, below_image_path, species_id, scenario,
  subset, confusing_pair_with, notes
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.comparative_metrics import (
    compute_accuracy,
    compute_coverage,
    evaluate_agreement,
    unified_outperforms_all,
    unified_wrong_but_standalone_right,
)
from benchmarks.comparative_reports import (
    generate_csv_report,
    generate_json_report,
    generate_markdown_report,
)
from benchmarks.config import PROJECT_ROOT
from benchmarks.manifest import ManifestDataset
from benchmarks.runners.base import RunnerResult
from benchmarks.key_tree_oracle import OracleKeyTree
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.llm_standalone_runner import LLMStandaloneRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.tree_runner import TreeRunner
from benchmarks.runners.unified_runner import UnifiedRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class _SimpleSample:
    """Minimal adapter so single-photo runners can consume a BenchmarkSpecimen."""

    def __init__(self, image_bytes: bytes, species_id: str):
        self.image_bytes = image_bytes
        self.species_id = species_id


def _run_method(
    method_name: str,
    runner,
    specimens: List[Any],
    single_photo: bool = True,
) -> List[RunnerResult]:
    """Run a single method on every specimen, logging progress."""
    results: List[RunnerResult] = []
    for i, spec in enumerate(specimens, 1):
        try:
            if single_photo:
                photo_bytes = spec.load_above_bytes() or spec.load_below_bytes()
                if photo_bytes is None:
                    results.append(
                        RunnerResult(
                            method_name=method_name,
                            predictions=[],
                            coverage=False,
                            error="No image available",
                        )
                    )
                    continue
                sample = _SimpleSample(image_bytes=photo_bytes, species_id=spec.species_id)
                result = runner.predict(sample)
            else:
                result = runner.predict(spec)
            results.append(result)
        except Exception as exc:
            logger.warning("%s failed on %s: %s", method_name, spec.specimen_id, exc)
            results.append(
                RunnerResult(
                    method_name=method_name,
                    predictions=[],
                    coverage=False,
                    error=str(exc),
                )
            )
        if i % 10 == 0 or i == len(specimens):
            logger.info("%s: %d/%d completed", method_name, i, len(specimens))
    return results


def _build_per_specimen_record(
    specimen,
    cnn_res: RunnerResult,
    tree_res: RunnerResult,
    db_res: RunnerResult,
    llm_res: RunnerResult,
    unified_res: RunnerResult,
) -> Dict[str, Any]:
    """Assemble the comparison dict for one specimen."""
    gt = specimen.species_id

    def _result_dict(r: RunnerResult) -> Dict[str, Any]:
        return {
            "top_species": r.top_species if r.coverage else "N/A",
            "confidence": r.top_confidence if r.coverage else 0.0,
            "coverage": r.coverage,
            "correct": r.coverage and r.top_species == gt,
            "error": r.error,
            "reasoning": r.metadata.get("llm_reasoning", "") if r.metadata else "",
        }

    agreement = evaluate_agreement(cnn_res, tree_res, db_res, unified_res)

    return {
        "specimen_id": specimen.specimen_id,
        "species_id": gt,
        "scenario": specimen.scenario,
        "subset": specimen.subset,
        "confusing_pair_with": specimen.confusing_pair_with,
        "notes": specimen.notes,
        "agreement": agreement,
        "results": {
            "cnn": _result_dict(cnn_res),
            "tree": _result_dict(tree_res),
            "db": _result_dict(db_res),
            "llm": _result_dict(llm_res),
            "unified": _result_dict(unified_res),
        },
        "unified_outperforms_all": unified_outperforms_all(
            unified_res, cnn_res, tree_res, db_res, llm_res, gt
        ),
        "unified_wrong_but_standalone_right": unified_wrong_but_standalone_right(
            unified_res, cnn_res, tree_res, db_res, llm_res, gt
        ),
    }


def _compute_metrics(
    cnn_results: List[RunnerResult],
    tree_results: List[RunnerResult],
    db_results: List[RunnerResult],
    llm_results: List[RunnerResult],
    unified_results: List[RunnerResult],
    specimens: List[Any],
    per_specimen: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate all comparative metrics."""
    ground_truth = [s.species_id for s in specimens]

    # Per-method overall
    per_method: Dict[str, Dict[str, Any]] = {}
    for name, results in (
        ("cnn", cnn_results),
        ("tree", tree_results),
        ("db", db_results),
        ("llm", llm_results),
        ("unified", unified_results),
    ):
        times = [r.inference_time_ms for r in results]
        per_method[name] = {
            "accuracy": compute_accuracy(results, ground_truth),
            "coverage": compute_coverage(results),
            "mean_time_ms": sum(times) / len(times) if times else 0.0,
        }

    # Agreement statistics
    agreement_counts: Dict[str, Dict[str, Any]] = {
        "agree": {"count": 0, "unified_correct": 0},
        "partial": {"count": 0, "unified_correct": 0},
        "disagree": {"count": 0, "unified_correct": 0},
        "inconclusive": {"count": 0, "unified_correct": 0},
    }
    for entry in per_specimen:
        level = entry["agreement"]
        if level not in agreement_counts:
            level = "inconclusive"
        agreement_counts[level]["count"] += 1
        if entry["results"]["unified"]["correct"]:
            agreement_counts[level]["unified_correct"] += 1

    total = len(per_specimen)
    agreement_stats: Dict[str, Dict[str, Any]] = {}
    for level, s in agreement_counts.items():
        agreement_stats[level] = {
            "count": s["count"],
            "pct": s["count"] / total if total else 0.0,
            "unified_accuracy": s["unified_correct"] / s["count"] if s["count"] else 0.0,
        }

    # By-scenario accuracy
    by_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario in ManifestDataset.SCENARIOS:
        indices = [i for i, s in enumerate(specimens) if s.scenario == scenario]
        if not indices:
            continue
        n = len(indices)
        by_scenario[scenario] = {
            "n": n,
            "cnn_acc": compute_accuracy([cnn_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "tree_acc": compute_accuracy([tree_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "db_acc": compute_accuracy([db_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "llm_acc": compute_accuracy([llm_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "unified_acc": compute_accuracy([unified_results[i] for i in indices], [ground_truth[i] for i in indices]),
        }

    # Confusing-pair breakdown
    confusing: Dict[str, Dict[str, Any]] = {}
    confusing_entries = [e for e in per_specimen if e["scenario"] == "confusing"]
    pair_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in confusing_entries:
        pair_key = e["confusing_pair_with"]
        if pair_key:
            canonical = "-".join(sorted([e["species_id"], pair_key]))
            pair_groups[canonical].append(e)

    for pair_name, entries in pair_groups.items():
        n = len(entries)
        cnn_c = sum(1 for e in entries if e["results"]["cnn"]["correct"])
        tree_c = sum(1 for e in entries if e["results"]["tree"]["correct"])
        db_c = sum(1 for e in entries if e["results"]["db"]["correct"])
        llm_c = sum(1 for e in entries if e["results"]["llm"]["correct"])
        uni_c = sum(1 for e in entries if e["results"]["unified"]["correct"])
        agr = sum(1 for e in entries if e["agreement"] in ("agree", "partial"))
        confusing[pair_name] = {
            "n": n,
            "cnn_acc": cnn_c / n if n else 0.0,
            "tree_acc": tree_c / n if n else 0.0,
            "db_acc": db_c / n if n else 0.0,
            "llm_acc": llm_c / n if n else 0.0,
            "unified_acc": uni_c / n if n else 0.0,
            "agreement_rate": agr / n if n else 0.0,
        }

    return {
        "per_method": per_method,
        "agreement_stats": agreement_stats,
        "by_scenario": by_scenario,
        "confusing_pairs": confusing,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run comparative benchmark across all identification methods"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv",
        help="Path to evaluation manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks" / "comparative",
        help="Directory to write reports",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cnn", "tree", "db", "llm", "unified", "all"],
        default=["all"],
        help="Methods to benchmark",
    )
    parser.add_argument(
        "--oracle-mode",
        action="store_true",
        help="Enable key-tree oracle: perfect pre-answers for 5 species, none for 5 others",
    )
    args = parser.parse_args()

    oracle = None
    if args.oracle_mode:
        oracle = OracleKeyTree(str(PROJECT_ROOT / "data" / "raw" / "key.xml"))
        logger.info("Oracle mode enabled. Oracle species (n=%d): %s",
                    len(oracle.oracle_species_ids), ", ".join(oracle.oracle_species_ids))
        logger.info("Non-oracle species (n=%d): %s",
                    len(oracle.non_oracle_species_ids), ", ".join(oracle.non_oracle_species_ids))

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ManifestDataset(args.manifest)
    specimens = list(dataset)
    logger.info("Loaded %d specimens from %s", len(specimens), args.manifest)

    if not specimens:
        raise ValueError("Manifest contains no specimens.")

    selected = set(args.methods)
    if "all" in selected:
        selected = {"cnn", "tree", "db", "llm", "unified"}

    cnn_results: List[RunnerResult] = []
    tree_results: List[RunnerResult] = []
    db_results: List[RunnerResult] = []
    llm_results: List[RunnerResult] = []
    unified_results: List[RunnerResult] = []

    if "cnn" in selected:
        logger.info("Initialising CNN runner...")
        cnn_results = _run_method("cnn", CNNRunner(), specimens, single_photo=True)

    if "tree" in selected:
        logger.info("Initialising Tree runner...")
        tree_results = _run_method("tree", TreeRunner(mode="auto", oracle=oracle), specimens, single_photo=True)

    if "db" in selected:
        logger.info("Initialising Trait-DB runner...")
        db_results = _run_method("db", TraitDBRunner(), specimens, single_photo=True)

    if "llm" in selected:
        logger.info("Initialising Standalone LLM runner...")
        llm_results = _run_method("llm", LLMStandaloneRunner(), specimens, single_photo=False)

    if "unified" in selected:
        logger.info("Initialising Unified runner...")
        unified_results = _run_method("unified", UnifiedRunner(oracle=oracle), specimens, single_photo=False)

    def _not_selected_results(method_name: str) -> List[RunnerResult]:
        return [
            RunnerResult(
                method_name=method_name,
                predictions=[],
                coverage=False,
                error="Method not selected",
            )
            for _ in specimens
        ]

    if not cnn_results:
        cnn_results = _not_selected_results("cnn")
    if not tree_results:
        tree_results = _not_selected_results("tree")
    if not db_results:
        db_results = _not_selected_results("db")
    if not llm_results:
        llm_results = _not_selected_results("llm")
    if not unified_results:
        unified_results = _not_selected_results("unified")

    per_specimen: List[Dict[str, Any]] = []
    for i, spec in enumerate(specimens):
        per_specimen.append(
            _build_per_specimen_record(
                spec,
                cnn_results[i],
                tree_results[i],
                db_results[i],
                llm_results[i],
                unified_results[i],
            )
        )

    metrics = _compute_metrics(
        cnn_results, tree_results, db_results, llm_results, unified_results,
        specimens, per_specimen,
    )

    generate_json_report(per_specimen, metrics, args.output_dir / "report.json")
    generate_csv_report(per_specimen, args.output_dir / "report.csv")
    generate_markdown_report(per_specimen, metrics, args.output_dir / "report.md")

    logger.info("Benchmark complete. Reports written to %s", args.output_dir)

    print("\n=== System A — Standalone Methods ===")
    for method, m in metrics["per_method"].items():
        if method in ("cnn", "tree", "db", "llm"):
            print(f"  {method:10s}: accuracy={m['accuracy']:.1%}, coverage={m['coverage']:.1%}")
    print("\n=== System B — Unified LLM Synthesis ===")
    for method, m in metrics["per_method"].items():
        if method == "unified":
            print(f"  {method:10s}: accuracy={m['accuracy']:.1%}, coverage={m['coverage']:.1%}")
    print("\n=== Raw Accuracy Comparison ===")
    unified_acc = metrics["per_method"].get("unified", {}).get("accuracy", 0)
    for method, m in metrics["per_method"].items():
        if method in ("cnn", "tree", "db", "llm"):
            diff = unified_acc - m["accuracy"]
            sign = "+" if diff >= 0 else ""
            print(f"  unified - {method:4s}: {sign}{diff:+.1%} (U={unified_acc:.1%}, {method}={m['accuracy']:.1%})")


if __name__ == "__main__":
    main()
