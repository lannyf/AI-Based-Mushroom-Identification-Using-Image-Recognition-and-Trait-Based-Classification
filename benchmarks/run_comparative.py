#!/usr/bin/env python3
"""Comparative benchmark runner for the unified mushroom identification pipeline.

Evaluates CNN reference, System A (A1 sub-tests), System A2 (A2 sub-tests),
and System B (B1/B2) on the same specimens and produces side-by-side
comparison reports.

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
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
from benchmarks.runners.tree_runner import TreeRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.unified_runner import UnifiedRunner
from benchmarks.species_trait_oracle import SpeciesTraitOracle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class _SimpleSample:
    """Minimal adapter so single-photo runners can consume a BenchmarkSpecimen."""

    def __init__(self, image_bytes: bytes, species_id: str):
        self.image_bytes = image_bytes
        self.species_id = species_id


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: Path) -> Dict[str, List[RunnerResult]]:
    """Load completed method results from checkpoint file."""
    if not checkpoint_path.exists():
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Rehydrate RunnerResult objects
        results: Dict[str, List[RunnerResult]] = {}
        for method_name, raw_results in data.items():
            results[method_name] = [
                RunnerResult(
                    method_name=r["method_name"],
                    predictions=[(p[0], p[1]) for p in r["predictions"]],
                    coverage=r["coverage"],
                    inference_time_ms=r["inference_time_ms"],
                    error=r.get("error"),
                    metadata=r.get("metadata", {}),
                )
                for r in raw_results
            ]
        logger.info("Loaded checkpoint with %d completed methods", len(results))
        return results
    except Exception as exc:
        logger.warning("Failed to load checkpoint: %s", exc)
        return {}


def _save_checkpoint(checkpoint_path: Path, all_results: Dict[str, List[RunnerResult]]) -> None:
    """Save all completed method results to checkpoint file."""
    serializable: Dict[str, List[Dict[str, Any]]] = {}
    for method_name, results in all_results.items():
        serializable[method_name] = [
            {
                "method_name": r.method_name,
                "predictions": r.predictions,
                "coverage": r.coverage,
                "inference_time_ms": r.inference_time_ms,
                "error": r.error,
                "metadata": r.metadata,
            }
            for r in results
        ]
    try:
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
    except Exception as exc:
        logger.warning("Failed to save checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# Specimen runner
# ---------------------------------------------------------------------------

def _run_one_specimen(
    method_name: str,
    runner,
    spec: Any,
    single_photo: bool,
    dual_photo_best: bool,
) -> RunnerResult:
    """Run a method on a single specimen."""
    try:
        if single_photo:
            above_bytes = spec.load_above_bytes()
            below_bytes = spec.load_below_bytes()

            if above_bytes is None and below_bytes is None:
                return RunnerResult(
                    method_name=method_name,
                    predictions=[],
                    coverage=False,
                    error="No image available",
                )

            if dual_photo_best and above_bytes and below_bytes:
                best_result = None
                best_conf = -1.0
                for photo_bytes in (above_bytes, below_bytes):
                    sample = _SimpleSample(
                        image_bytes=photo_bytes, species_id=spec.species_id
                    )
                    res = runner.predict(sample)
                    top_conf = res.top_confidence if res.coverage else -1.0
                    if top_conf > best_conf:
                        best_conf = top_conf
                        best_result = res
                return best_result  # type: ignore[return-value]
            else:
                photo_bytes = above_bytes or below_bytes
                sample = _SimpleSample(
                    image_bytes=photo_bytes, species_id=spec.species_id
                )
                return runner.predict(sample)
        else:
            return runner.predict(spec)
    except Exception as exc:
        logger.warning("%s failed on %s: %s", method_name, spec.specimen_id, exc)
        return RunnerResult(
            method_name=method_name,
            predictions=[],
            coverage=False,
            error=str(exc),
        )


def _run_method(
    method_name: str,
    runner,
    specimens: List[Any],
    single_photo: bool = True,
    dual_photo_best: bool = False,
    workers: int = 1,
) -> List[RunnerResult]:
    """Run a single method on every specimen, logging progress.

    Args:
        method_name: Identifier for logging.
        runner: The benchmark runner instance.
        specimens: List of BenchmarkSpecimen objects.
        single_photo: If True, the runner expects a single image.
        dual_photo_best: If True *and* single_photo is True, run the runner on
            both above and below photos and keep the result with the higher
            top-1 confidence.
        workers: Number of parallel threads for specimen processing.
            Values > 1 are useful for CPU/GPU-bound runners (e.g. CNN).
            LLM-based runners should use workers=1.
    """
    results: List[RunnerResult] = [None] * len(specimens)  # type: ignore[list-item]

    if workers > 1 and single_photo:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_one_specimen,
                    method_name,
                    runner,
                    spec,
                    single_photo,
                    dual_photo_best,
                ): idx
                for idx, spec in enumerate(specimens)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(specimens):
                    logger.info(
                        "%s: %d/%d completed", method_name, completed, len(specimens)
                    )
    else:
        for i, spec in enumerate(specimens, 1):
            results[i - 1] = _run_one_specimen(
                method_name, runner, spec, single_photo, dual_photo_best
            )
            if i % 10 == 0 or i == len(specimens):
                logger.info("%s: %d/%d completed", method_name, i, len(specimens))
    return results


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _build_per_specimen_record(
    specimen,
    cnn_res: RunnerResult,
    a1_vision_res: RunnerResult,
    a1_llm_res: RunnerResult,
    a1_tree_res: RunnerResult,
    a1_db_res: RunnerResult,
    a2_llm_res: RunnerResult,
    a2_tree_res: RunnerResult,
    a2_db_res: RunnerResult,
    b1_res: RunnerResult,
    b2_res: RunnerResult,
) -> Dict[str, Any]:
    """Assemble the comparison dict for one specimen."""
    gt = specimen.species_id

    def _result_dict(r: RunnerResult) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "top_species": r.top_species if r.coverage else "N/A",
            "confidence": r.top_confidence if r.coverage else 0.0,
            "coverage": r.coverage,
            "correct": r.coverage and r.top_species == gt,
            "error": r.error,
            "reasoning": r.metadata.get("llm_reasoning", "") if r.metadata else "",
        }
        if r.metadata:
            d["signals"] = {
                "llm_raw": r.metadata.get("llm_raw_species"),
                "llm_confidence": r.metadata.get("llm_confidence"),
                "cnn": r.metadata.get("cnn_pred"),
                "cnn_confidence": r.metadata.get("cnn_confidence"),
                "tree": r.metadata.get("tree_pred"),
                "db": r.metadata.get("db_pred"),
                "db_species_id": r.metadata.get("db_species_id"),
                "db_score": r.metadata.get("db_score"),
                "final_recommendation": r.metadata.get("final_recommendation"),
                "used_fallback": r.metadata.get("used_fallback", False),
                "agreement": r.metadata.get("agreement"),
            }
        return d

    return {
        "specimen_id": specimen.specimen_id,
        "species_id": gt,
        "scenario": specimen.scenario,
        "subset": specimen.subset,
        "confusing_pair_with": specimen.confusing_pair_with,
        "notes": specimen.notes,
        "results": {
            "cnn": _result_dict(cnn_res),
            "a1_vision": _result_dict(a1_vision_res),
            "a1_llm": _result_dict(a1_llm_res),
            "a1_tree": _result_dict(a1_tree_res),
            "a1_db": _result_dict(a1_db_res),
            "a2_llm": _result_dict(a2_llm_res),
            "a2_tree": _result_dict(a2_tree_res),
            "a2_db": _result_dict(a2_db_res),
            "b1": _result_dict(b1_res),
            "b2": _result_dict(b2_res),
        },
    }


def _compute_metrics(
    cnn_results: List[RunnerResult],
    a1_vision_results: List[RunnerResult],
    a1_llm_results: List[RunnerResult],
    a1_tree_results: List[RunnerResult],
    a1_db_results: List[RunnerResult],
    a2_llm_results: List[RunnerResult],
    a2_tree_results: List[RunnerResult],
    a2_db_results: List[RunnerResult],
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
        ("a1_vision", a1_vision_results),
        ("a1_llm", a1_llm_results),
        ("a1_tree", a1_tree_results),
        ("a1_db", a1_db_results),
        ("a2_llm", a2_llm_results),
        ("a2_tree", a2_tree_results),
        ("a2_db", a2_db_results),
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
            "a1_vision_acc": compute_accuracy([a1_vision_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a1_llm_acc": compute_accuracy([a1_llm_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a1_tree_acc": compute_accuracy([a1_tree_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a1_db_acc": compute_accuracy([a1_db_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a2_llm_acc": compute_accuracy([a2_llm_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a2_tree_acc": compute_accuracy([a2_tree_results[i] for i in indices], [ground_truth[i] for i in indices]),
            "a2_db_acc": compute_accuracy([a2_db_results[i] for i in indices], [ground_truth[i] for i in indices]),
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
        confusing[pair_name] = {
            "n": n,
            "cnn_acc": sum(1 for e in entries if e["results"]["cnn"]["correct"]) / n if n else 0.0,
            "a1_vision_acc": sum(1 for e in entries if e["results"]["a1_vision"]["correct"]) / n if n else 0.0,
            "a1_llm_acc": sum(1 for e in entries if e["results"]["a1_llm"]["correct"]) / n if n else 0.0,
            "a1_tree_acc": sum(1 for e in entries if e["results"]["a1_tree"]["correct"]) / n if n else 0.0,
            "a1_db_acc": sum(1 for e in entries if e["results"]["a1_db"]["correct"]) / n if n else 0.0,
            "a2_llm_acc": sum(1 for e in entries if e["results"]["a2_llm"]["correct"]) / n if n else 0.0,
            "a2_tree_acc": sum(1 for e in entries if e["results"]["a2_tree"]["correct"]) / n if n else 0.0,
            "a2_db_acc": sum(1 for e in entries if e["results"]["a2_db"]["correct"]) / n if n else 0.0,
            "b1_acc": sum(1 for e in entries if e["results"]["b1"]["correct"]) / n if n else 0.0,
            "b2_acc": sum(1 for e in entries if e["results"]["b2"]["correct"]) / n if n else 0.0,
        }

    # Oracle benefit per component
    oracle_benefit_llm = per_method["a2_llm"]["accuracy"] - per_method["a1_llm"]["accuracy"]
    oracle_benefit_tree = per_method["a2_tree"]["accuracy"] - per_method["a1_tree"]["accuracy"]
    oracle_benefit_db = per_method["a2_db"]["accuracy"] - per_method["a1_db"]["accuracy"]

    # Synthesis value: does B beat the best standalone in its system?
    best_a1 = max(
        per_method["a1_vision"]["accuracy"],
        per_method["a1_llm"]["accuracy"],
        per_method["a1_tree"]["accuracy"],
        per_method["a1_db"]["accuracy"],
    )
    best_a2 = max(
        per_method["a2_llm"]["accuracy"],
        per_method["a2_tree"]["accuracy"],
        per_method["a2_db"]["accuracy"],
    )
    synthesis_benefit_b1 = per_method["b1"]["accuracy"] - best_a1
    synthesis_benefit_b2 = per_method["b2"]["accuracy"] - best_a2
    extractor_penalty = per_method["b1"]["accuracy"] - per_method["b2"]["accuracy"]

    return {
        "per_method": per_method,
        "by_scenario": by_scenario,
        "confusing_pairs": confusing,
        "oracle_benefit_llm": oracle_benefit_llm,
        "oracle_benefit_tree": oracle_benefit_tree,
        "oracle_benefit_db": oracle_benefit_db,
        "best_a1": best_a1,
        "best_a2": best_a2,
        "synthesis_benefit_b1": synthesis_benefit_b1,
        "synthesis_benefit_b2": synthesis_benefit_b2,
        "extractor_penalty": extractor_penalty,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        choices=[
            "cnn",
            "a1_vision", "a1_llm", "a1_tree", "a1_db",
            "a2_llm", "a2_tree", "a2_db",
            "b1", "b2",
            "all",
        ],
        default=["all"],
        help="Variants to benchmark",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional JSON checkpoint file to resume from",
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
        selected = {
            "cnn",
            "a1_vision", "a1_llm", "a1_tree", "a1_db",
            "a2_llm", "a2_tree", "a2_db",
            "b1", "b2",
        }

    # Initialise oracle (used by A2 and B2)
    oracle = SpeciesTraitOracle(str(PROJECT_ROOT / "data" / "raw" / "species_traits.xml"))

    # Load checkpoint if provided
    checkpoint_path = args.checkpoint or (args.output_dir / "checkpoint.json")
    checkpoint = _load_checkpoint(checkpoint_path)

    # Result containers
    all_results: Dict[str, List[RunnerResult]] = {}

    def _maybe_run(
        method_name: str,
        runner,
        single_photo: bool = True,
        dual_photo_best: bool = False,
        workers: int = 1,
    ) -> List[RunnerResult]:
        """Run a method if selected and not already in checkpoint."""
        if method_name not in selected:
            logger.info("Skipping %s (not selected)", method_name)
            return [
                RunnerResult(
                    method_name=method_name,
                    predictions=[],
                    coverage=False,
                    error="Variant not selected",
                )
                for _ in specimens
            ]
        if method_name in checkpoint:
            logger.info("Resuming %s from checkpoint (%d results)", method_name, len(checkpoint[method_name]))
            return checkpoint[method_name]

        logger.info("Running %s...", method_name)
        results = _run_method(
            method_name, runner, specimens,
            single_photo=single_photo,
            dual_photo_best=dual_photo_best,
            workers=workers,
        )
        all_results[method_name] = results
        _save_checkpoint(checkpoint_path, all_results)
        return results

    # Run CNN in background (parallel with LLM methods)
    cnn_future = None
    cnn_executor = None
    if "cnn" in selected and "cnn" not in checkpoint:
        logger.info("Initialising CNN runner...")
        cnn_executor = ThreadPoolExecutor(max_workers=1)
        cnn_future = cnn_executor.submit(
            _run_method,
            "cnn",
            CNNRunner(),
            specimens,
            single_photo=True,
            dual_photo_best=True,
            workers=4,
        )

    # System A1 — extracted traits
    a1_vision_results = _maybe_run("a1_vision", LLMStandaloneRunner(trait_source="none"), single_photo=False)
    a1_llm_results = _maybe_run("a1_llm", LLMStandaloneRunner(trait_source="extracted"), single_photo=False)
    a1_tree_results = _maybe_run("a1_tree", TreeRunner(trait_source="extracted"), single_photo=False)
    a1_db_results = _maybe_run("a1_db", TraitDBRunner(trait_source="extracted"), single_photo=False)

    # System A2 — oracle traits
    a2_llm_results = _maybe_run("a2_llm", LLMStandaloneRunner(trait_source="oracle", oracle_trait_provider=oracle), single_photo=False)
    a2_tree_results = _maybe_run("a2_tree", TreeRunner(trait_source="oracle", oracle_trait_provider=oracle), single_photo=False)
    a2_db_results = _maybe_run("a2_db", TraitDBRunner(trait_source="oracle", oracle_trait_provider=oracle), single_photo=False)

    # System B — unified pipeline
    b1_results = _maybe_run("b1", UnifiedRunner(oracle_trait_provider=None), single_photo=False)
    b2_results = _maybe_run("b2", UnifiedRunner(oracle_trait_provider=oracle), single_photo=False)

    # Wait for CNN
    if cnn_future is not None:
        cnn_results = cnn_future.result()
        all_results["cnn"] = cnn_results
        _save_checkpoint(checkpoint_path, all_results)
        if cnn_executor:
            cnn_executor.shutdown(wait=False)
    else:
        cnn_results = checkpoint.get("cnn", _maybe_run("cnn", CNNRunner(), single_photo=True, dual_photo_best=True, workers=4))

    # Assemble per-specimen records
    per_specimen: List[Dict[str, Any]] = []
    for i, spec in enumerate(specimens):
        per_specimen.append(
            _build_per_specimen_record(
                spec,
                cnn_results[i],
                a1_vision_results[i],
                a1_llm_results[i],
                a1_tree_results[i],
                a1_db_results[i],
                a2_llm_results[i],
                a2_tree_results[i],
                a2_db_results[i],
                b1_results[i],
                b2_results[i],
            )
        )

    metrics = _compute_metrics(
        cnn_results,
        a1_vision_results, a1_llm_results, a1_tree_results, a1_db_results,
        a2_llm_results, a2_tree_results, a2_db_results,
        b1_results, b2_results,
        specimens, per_specimen,
    )

    generate_json_report(per_specimen, metrics, args.output_dir / "report.json")
    generate_csv_report(per_specimen, metrics, args.output_dir / "report.csv")
    generate_markdown_report(per_specimen, metrics, args.output_dir / "report.md")

    logger.info("Benchmark complete. Reports written to %s", args.output_dir)

    # Print summary
    print("\n=== Overall Accuracy ===")
    for method, m in metrics["per_method"].items():
        print(f"  {method:15s}: accuracy={m['accuracy']:.1%}, coverage={m['coverage']:.1%}")

    print("\n=== Oracle Benefit (A2 − A1) ===")
    print(f"  LLM:  {metrics['oracle_benefit_llm']:+.1%}")
    print(f"  Tree: {metrics['oracle_benefit_tree']:+.1%}")
    print(f"  DB:   {metrics['oracle_benefit_db']:+.1%}")

    print("\n=== Synthesis Benefit (B − best standalone) ===")
    print(f"  B1 − best A1: {metrics['synthesis_benefit_b1']:+.1%}")
    print(f"  B2 − best A2: {metrics['synthesis_benefit_b2']:+.1%}")
    print(f"  B1 − B2 (extractor penalty): {metrics['extractor_penalty']:+.1%}")


if __name__ == "__main__":
    main()
