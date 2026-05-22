#!/usr/bin/env python3
"""Comparative benchmark runner for the unified mushroom identification pipeline.

Evaluates CNN reference, System A (A1/A2), and System B (B1/B2) on the same
specimens and produces side-by-side comparison reports.

Usage::

    python -m benchmarks.run_comparative \
        --manifest benchmarks/evaluation_manifest_v2.csv \
        --variants all \
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
)
from benchmarks.comparative_reports import (
    generate_csv_report,
    generate_json_report,
    generate_markdown_report,
)
from benchmarks.config import PROJECT_ROOT
from benchmarks.manifest import ManifestDataset
from benchmarks.runners.base import RunnerResult
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.llm_standalone_runner import LLMStandaloneRunner
from benchmarks.runners.unified_runner import UnifiedRunner
from benchmarks.species_trait_oracle import SpeciesTraitOracle

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
    a1_res: RunnerResult,
    a2_res: RunnerResult,
    b1_res: RunnerResult,
    b2_res: RunnerResult,
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

    return {
        "specimen_id": specimen.specimen_id,
        "species_id": gt,
        "scenario": specimen.scenario,
        "subset": specimen.subset,
        "confusing_pair_with": specimen.confusing_pair_with,
        "notes": specimen.notes,
        "results": {
            "cnn": _result_dict(cnn_res),
            "a1": _result_dict(a1_res),
            "a2": _result_dict(a2_res),
            "b1": _result_dict(b1_res),
            "b2": _result_dict(b2_res),
        },
    }


def _compute_metrics(
    cnn_results: List[RunnerResult],
    a1_results: List[RunnerResult],
    a2_results: List[RunnerResult],
    b1_results: List[RunnerResult],
    b2_results: List[RunnerResult],
    specimens: List[Any],
    per_specimen: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate all comparative metrics."""
    ground_truth = [s.species_id for s in specimens]

    # Per-method overall
    per_method: Dict[str, Dict[str, Any]] = {}
    for name, results in (
        ("cnn", cnn_results),
        ("a1", a1_results),
        ("a2", a2_results),
        ("b1", b1_results),
        ("b2", b2_results),
    ):
        times = [r.inference_time_ms for r in results]
        per_method[name] = {
            "accuracy": compute_accuracy(results, ground_truth),
            "coverage": compute_coverage(results),
            "mean_time_ms": sum(times) / len(times) if times else 0.0,
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
            "a1_acc": compute_accuracy([a1_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a2_acc": compute_accuracy([a2_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "b1_acc": compute_accuracy([b1_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "b2_acc": compute_accuracy([b2_results[i] for i in indices], [ground_truth[i] for i in indices]),
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
        a1_c = sum(1 for e in entries if e["results"]["a1"]["correct"])
        a2_c = sum(1 for e in entries if e["results"]["a2"]["correct"])
        b1_c = sum(1 for e in entries if e["results"]["b1"]["correct"])
        b2_c = sum(1 for e in entries if e["results"]["b2"]["correct"])
        confusing[pair_name] = {
            "n": n,
            "cnn_acc": cnn_c / n if n else 0.0,
            "a1_acc": a1_c / n if n else 0.0,
            "a2_acc": a2_c / n if n else 0.0,
            "b1_acc": b1_c / n if n else 0.0,
            "b2_acc": b2_c / n if n else 0.0,
        }

    # Oracle impact deltas
    extractor_penalty = per_method["b1"]["accuracy"] - per_method["b2"]["accuracy"]
    oracle_benefit_a = per_method["a2"]["accuracy"] - per_method["a1"]["accuracy"]
    oracle_benefit_b = per_method["b2"]["accuracy"] - per_method["b1"]["accuracy"]

    return {
        "per_method": per_method,
        "by_scenario": by_scenario,
        "confusing_pairs": confusing,
        "extractor_penalty": extractor_penalty,
        "oracle_benefit_a": oracle_benefit_a,
        "oracle_benefit_b": oracle_benefit_b,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run comparative benchmark across all identification variants"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "evaluation_manifest_v2.csv",
        help="Path to evaluation manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks" / "comparative",
        help="Directory to write reports",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["cnn", "a1", "a2", "b1", "b2", "all"],
        default=["all"],
        help="Variants to benchmark",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ManifestDataset(args.manifest)
    specimens = list(dataset)
    logger.info("Loaded %d specimens from %s", len(specimens), args.manifest)

    if not specimens:
        raise ValueError("Manifest contains no specimens.")

    selected = set(args.variants)
    if "all" in selected:
        selected = {"cnn", "a1", "a2", "b1", "b2"}

    # Initialise oracle (used by A2 and B2)
    oracle = SpeciesTraitOracle(str(PROJECT_ROOT / "data" / "raw" / "species_traits.xml"))

    cnn_results: List[RunnerResult] = []
    a1_results: List[RunnerResult] = []
    a2_results: List[RunnerResult] = []
    b1_results: List[RunnerResult] = []
    b2_results: List[RunnerResult] = []

    if "cnn" in selected:
        logger.info("Initialising CNN runner...")
        cnn_results = _run_method("cnn", CNNRunner(), specimens, single_photo=True)

    if "a1" in selected:
        logger.info("Initialising A1 runner (LLM raw)...")
        a1_results = _run_method("a1", LLMStandaloneRunner(oracle_trait_provider=None), specimens, single_photo=False)

    if "a2" in selected:
        logger.info("Initialising A2 runner (LLM + oracle traits)...")
        a2_results = _run_method("a2", LLMStandaloneRunner(oracle_trait_provider=oracle), specimens, single_photo=False)

    if "b1" in selected:
        logger.info("Initialising B1 runner (Unified)...")
        b1_results = _run_method("b1", UnifiedRunner(oracle_trait_provider=None), specimens, single_photo=False)

    if "b2" in selected:
        logger.info("Initialising B2 runner (Unified + oracle traits)...")
        b2_results = _run_method("b2", UnifiedRunner(oracle_trait_provider=oracle), specimens, single_photo=False)

    def _not_selected_results(method_name: str) -> List[RunnerResult]:
        return [
            RunnerResult(
                method_name=method_name,
                predictions=[],
                coverage=False,
                error="Variant not selected",
            )
            for _ in specimens
        ]

    if not cnn_results:
        cnn_results = _not_selected_results("cnn")
    if not a1_results:
        a1_results = _not_selected_results("a1")
    if not a2_results:
        a2_results = _not_selected_results("a2")
    if not b1_results:
        b1_results = _not_selected_results("b1")
    if not b2_results:
        b2_results = _not_selected_results("b2")

    per_specimen: List[Dict[str, Any]] = []
    for i, spec in enumerate(specimens):
        per_specimen.append(
            _build_per_specimen_record(
                spec,
                cnn_results[i],
                a1_results[i],
                a2_results[i],
                b1_results[i],
                b2_results[i],
            )
        )

    metrics = _compute_metrics(
        cnn_results, a1_results, a2_results, b1_results, b2_results,
        specimens, per_specimen,
    )

    generate_json_report(per_specimen, metrics, args.output_dir / "report.json")
    generate_csv_report(per_specimen, args.output_dir / "report.csv")
    generate_markdown_report(per_specimen, metrics, args.output_dir / "report.md")

    logger.info("Benchmark complete. Reports written to %s", args.output_dir)

    print("\n=== Overall Accuracy ===")
    for method, m in metrics["per_method"].items():
        print(f"  {method:10s}: accuracy={m['accuracy']:.1%}, coverage={m['coverage']:.1%}")

    print("\n=== Oracle Impact ===")
    print(f"  A2 - A1 (raw LLM benefit from oracle traits): {metrics['oracle_benefit_a']:+.1%}")
    print(f"  B2 - B1 (unified benefit from perfect traits): {metrics['oracle_benefit_b']:+.1%}")
    print(f"  B1 - B2 (extractor penalty): {metrics['extractor_penalty']:+.1%}")


if __name__ == "__main__":
    main()
