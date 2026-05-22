"""Report generators for the comparative benchmark.

Produces three outputs from a completed comparative run:
  * JSON — full structured data for downstream analysis.
  * CSV  — one row per specimen, one column triplet per method.
  * Markdown — thesis-ready tables with System A / System B separation.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.runners.base import RunnerResult


_ALL_VARIANTS = ["cnn", "a1", "a2", "b1", "b2"]


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def generate_json_report(
    per_specimen: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write the complete per-specimen breakdown and summary metrics as JSON."""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_specimens": len(per_specimen),
        },
        "metrics": metrics,
        "per_specimen": per_specimen,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------


def generate_csv_report(
    per_specimen: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a flat CSV with one row per specimen.

    Each method contributes ``{method}_pred``, ``{method}_correct``,
    ``{method}_coverage``, and ``{method}_confidence`` columns.
    """
    if not per_specimen:
        return

    # Infer method names from the first specimen's results dict
    methods = list(per_specimen[0]["results"].keys())

    fieldnames = [
        "specimen_id",
        "species_id",
        "scenario",
        "subset",
        "confusing_pair_with",
    ]
    for m in methods:
        fieldnames.extend([f"{m}_pred", f"{m}_correct", f"{m}_coverage", f"{m}_confidence"])
    fieldnames.append("notes")

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in per_specimen:
            row: Dict[str, Any] = {
                "specimen_id": entry["specimen_id"],
                "species_id": entry["species_id"],
                "scenario": entry["scenario"],
                "subset": entry["subset"],
                "confusing_pair_with": entry.get("confusing_pair_with", ""),
                "notes": entry.get("notes", ""),
            }
            for m, r in entry["results"].items():
                row[f"{m}_pred"] = r.get("top_species", "N/A")
                row[f"{m}_correct"] = "1" if r.get("correct", False) else "0"
                row[f"{m}_coverage"] = "1" if r.get("coverage", False) else "0"
                row[f"{m}_confidence"] = r.get("confidence", 0.0)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Markdown report (thesis-ready)
# ---------------------------------------------------------------------------


def generate_markdown_report(
    per_specimen: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write a thesis-ready Markdown report with 5 tables."""
    lines: List[str] = ["# Comparative Benchmark Results\n"]

    per_method = metrics.get("per_method", {})

    # --- 1. Overall Accuracy Table ----------------------------------------
    lines.append("## 1. Overall Accuracy\n")
    lines.append("| Method | Accuracy | Coverage | Mean Time (ms) |")
    lines.append("|--------|----------|----------|----------------|")
    for method in _ALL_VARIANTS:
        m = per_method.get(method, {})
        lines.append(
            f"| {method} | {m.get('accuracy', 0):.1%} | {m.get('coverage', 0):.1%} | {m.get('mean_time_ms', 0):.1f} |"
        )
    lines.append("")

    # --- 2. Accuracy by Scenario ------------------------------------------
    lines.append("## 2. Accuracy by Scenario\n")
    lines.append("| Scenario | N | CNN | A1 | A2 | B1 | B2 |")
    lines.append("|----------|---|-----|----|----|----|----|")
    for scenario, s in metrics.get("by_scenario", {}).items():
        lines.append(
            f"| {scenario} | {s['n']} | {s['cnn_acc']:.0%} | {s['a1_acc']:.0%} | "
            f"{s['a2_acc']:.0%} | {s['b1_acc']:.0%} | {s['b2_acc']:.0%} |"
        )
    lines.append("")

    # --- 3. Oracle Impact: A2 vs A1 ---------------------------------------
    lines.append("## 3. Oracle Impact: A2 vs A1\n")
    lines.append("*How much raw LLM benefits from perfect vision-only trait knowledge.*\n")
    a1_acc = per_method.get("a1", {}).get("accuracy", 0)
    a2_acc = per_method.get("a2", {}).get("accuracy", 0)
    delta_a = a2_acc - a1_acc
    lines.append(f"- A1 accuracy: {a1_acc:.1%}")
    lines.append(f"- A2 accuracy: {a2_acc:.1%}")
    lines.append(f"- Delta (A2 - A1): {delta_a:+.1%}\n")

    # --- 4. Trait Extractor Impact: B1 vs B2 ------------------------------
    lines.append("## 4. Trait Extractor Impact: B1 vs B2\n")
    lines.append(
        "*Performance lost to imperfect trait extraction. "
        "Positive extractor_penalty = B1 performs worse than B2.*\n"
    )
    b1_acc = per_method.get("b1", {}).get("accuracy", 0)
    b2_acc = per_method.get("b2", {}).get("accuracy", 0)
    penalty = metrics.get("extractor_penalty", b1_acc - b2_acc)
    lines.append(f"- B1 accuracy (extracted traits): {b1_acc:.1%}")
    lines.append(f"- B2 accuracy (oracle traits): {b2_acc:.1%}")
    lines.append(f"- Extractor penalty (B1 - B2): {penalty:+.1%}\n")

    # --- 5. Confusing Pair Breakdown --------------------------------------
    lines.append("## 5. Confusing Pair Breakdown\n")
    confusing = metrics.get("confusing_pairs", {})
    if confusing:
        lines.append("| Pair | N | CNN | A1 | A2 | B1 | B2 |")
        lines.append("|------|---|-----|----|----|----|----|")
        for pair_name, p in confusing.items():
            lines.append(
                f"| {pair_name} | {p['n']} | {p['cnn_acc']:.0%} | {p['a1_acc']:.0%} | "
                f"{p['a2_acc']:.0%} | {p['b1_acc']:.0%} | {p['b2_acc']:.0%} |"
            )
    else:
        lines.append("*No confusing-pair specimens were evaluated.*")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
