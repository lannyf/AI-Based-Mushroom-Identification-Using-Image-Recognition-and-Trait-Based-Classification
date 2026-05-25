"""Report generators for the comparative benchmark.

Produces three outputs from a completed comparative run:
  * JSON — full structured data for downstream analysis.
  * CSV  — one row per specimen, one column triplet per method.
  * Markdown — thesis-ready tables with System A1 / A2 / B separation.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.runners.base import RunnerResult


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
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write a flat CSV with one row per specimen.

    Each method contributes ``{method}_pred``, ``{method}_correct``,
    ``{method}_coverage``, and ``{method}_confidence`` columns.
    Unified methods (b1, b2) additionally emit ``{method}_llm_raw``,
    ``{method}_cnn``, ``{method}_tree``, and ``{method}_db`` columns
    so downstream analysis can see what each subsystem predicted.
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
        if m in ("b1", "b2"):
            fieldnames.extend([
                f"{m}_llm_raw",
                f"{m}_llm_conf",
                f"{m}_cnn",
                f"{m}_tree",
                f"{m}_db",
                f"{m}_db_sid",
                f"{m}_used_fallback",
            ])
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
                if m in ("b1", "b2"):
                    sig = r.get("signals", {}) or {}
                    row[f"{m}_llm_raw"] = sig.get("llm_raw", "")
                    row[f"{m}_llm_conf"] = sig.get("llm_confidence", "")
                    row[f"{m}_cnn"] = sig.get("cnn", "")
                    row[f"{m}_tree"] = sig.get("tree", "")
                    row[f"{m}_db"] = sig.get("db", "")
                    row[f"{m}_db_sid"] = sig.get("db_species_id", "")
                    row[f"{m}_used_fallback"] = "1" if sig.get("used_fallback") else "0"
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Markdown report (thesis-ready)
# ---------------------------------------------------------------------------

# Ordering used in all tables.
_A1_METHODS = ["a1_vision", "a1_llm", "a1_tree", "a1_db"]
_A2_METHODS = ["a2_llm", "a2_tree", "a2_db"]
_B_METHODS  = ["b1", "b2"]
_ALL_METHODS = ["cnn"] + _A1_METHODS + _A2_METHODS + _B_METHODS


def _fmt_acc(val: float) -> str:
    return f"{val:.1%}"


def generate_markdown_report(
    per_specimen: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write a thesis-ready Markdown report with focused tables."""
    lines: List[str] = ["# Comparative Benchmark Results\n"]
    per_method = metrics.get("per_method", {})

    # --- 1. Overall Accuracy (all methods) --------------------------------
    lines.append("## 1. Overall Accuracy\n")
    lines.append("| Method | Accuracy | Coverage | Mean Time (ms) |")
    lines.append("|--------|----------|----------|----------------|")
    for method in _ALL_METHODS:
        m = per_method.get(method, {})
        lines.append(
            f"| {method} | {m.get('accuracy', 0):.1%} | {m.get('coverage', 0):.1%} | {m.get('mean_time_ms', 0):.1f} |"
        )
    lines.append("")

    # --- 2. System A1: Component Breakdown --------------------------------
    lines.append("## 2. System A1 — Extracted-Trait Components\n")
    lines.append("| Method | Accuracy | Coverage |")
    lines.append("|--------|----------|----------|")
    for method in _A1_METHODS:
        m = per_method.get(method, {})
        lines.append(f"| {method} | {_fmt_acc(m.get('accuracy', 0))} | {_fmt_acc(m.get('coverage', 0))} |")
    lines.append("")

    # --- 3. System A2: Component Breakdown --------------------------------
    lines.append("## 3. System A2 — Oracle-Trait Components\n")
    lines.append("| Method | Accuracy | Coverage |")
    lines.append("|--------|----------|----------|")
    for method in _A2_METHODS:
        m = per_method.get(method, {})
        lines.append(f"| {method} | {_fmt_acc(m.get('accuracy', 0))} | {_fmt_acc(m.get('coverage', 0))} |")
    lines.append("")

    # --- 4. System B: Unified Pipeline ------------------------------------
    lines.append("## 4. System B — Unified Pipeline\n")
    lines.append("| Method | Accuracy | Coverage |")
    lines.append("|--------|----------|----------|")
    for method in _B_METHODS:
        m = per_method.get(method, {})
        lines.append(f"| {method} | {_fmt_acc(m.get('accuracy', 0))} | {_fmt_acc(m.get('coverage', 0))} |")
    lines.append("")

    # --- 5. Oracle Benefit per Component ----------------------------------
    lines.append("## 5. Oracle Benefit (A2 − A1) per Component\n")
    lines.append("*How much each standalone component gains from perfect trait knowledge.*\n")
    lines.append("| Component | A1 (extracted) | A2 (oracle) | Δ |")
    lines.append("|-----------|----------------|-------------|---|")
    for comp in ["llm", "tree", "db"]:
        a1 = per_method.get(f"a1_{comp}", {}).get("accuracy", 0)
        a2 = per_method.get(f"a2_{comp}", {}).get("accuracy", 0)
        delta = a2 - a1
        lines.append(f"| {comp} | {_fmt_acc(a1)} | {_fmt_acc(a2)} | {delta:+.1%} |")
    lines.append("")

    # --- 6. Synthesis Benefit ---------------------------------------------
    lines.append("## 6. Synthesis Benefit (B − best standalone)\n")
    lines.append("*Does the unified pipeline outperform the best standalone component in its system?*\n")
    best_a1 = metrics.get("best_a1", 0)
    best_a2 = metrics.get("best_a2", 0)
    b1 = per_method.get("b1", {}).get("accuracy", 0)
    b2 = per_method.get("b2", {}).get("accuracy", 0)
    lines.append(f"| System | Best Standalone | Unified | Δ (synthesis) |")
    lines.append(f"|--------|-----------------|---------|---------------|")
    lines.append(f"| A1 → B1 | {_fmt_acc(best_a1)} | {_fmt_acc(b1)} | {b1 - best_a1:+.1%} |")
    lines.append(f"| A2 → B2 | {_fmt_acc(best_a2)} | {_fmt_acc(b2)} | {b2 - best_a2:+.1%} |")
    lines.append("")

    # --- 7. Trait Extractor Impact ----------------------------------------
    lines.append("## 7. Trait Extractor Impact (B1 vs B2)\n")
    lines.append(
        "*Performance lost to imperfect automatic trait extraction. "
        "Positive extractor_penalty = B1 performs worse than B2.*\n"
    )
    penalty = metrics.get("extractor_penalty", b1 - b2)
    lines.append(f"- B1 accuracy (extracted traits): {_fmt_acc(b1)}")
    lines.append(f"- B2 accuracy (oracle traits): {_fmt_acc(b2)}")
    lines.append(f"- Extractor penalty (B1 − B2): {penalty:+.1%}\n")

    # --- 8. Accuracy by Scenario ------------------------------------------
    lines.append("## 8. Accuracy by Scenario\n")
    # Build dynamic header based on actual methods present
    methods_in_data = list(per_specimen[0]["results"].keys()) if per_specimen else _ALL_METHODS
    hdr_methods = " | ".join(methods_in_data)
    lines.append(f"| Scenario | N | {hdr_methods} |")
    sep = "|----------|---|" + "|".join(["-----"] * len(methods_in_data)) + "|"
    lines.append(sep)
    for scenario, s in metrics.get("by_scenario", {}).items():
        cells = " | ".join(_fmt_acc(s.get(f"{m}_acc", 0)) for m in methods_in_data)
        lines.append(f"| {scenario} | {s['n']} | {cells} |")
    lines.append("")

    # --- 9. Confusing Pair Breakdown --------------------------------------
    lines.append("## 9. Confusing Pair Breakdown\n")
    confusing = metrics.get("confusing_pairs", {})
    if confusing:
        lines.append(f"| Pair | N | {hdr_methods} |")
        sep = "|------|---|" + "|".join(["-----"] * len(methods_in_data)) + "|"
        lines.append(sep)
        for pair_name, p in confusing.items():
            cells = " | ".join(_fmt_acc(p.get(f"{m}_acc", 0)) for m in methods_in_data)
            lines.append(f"| {pair_name} | {p['n']} | {cells} |")
    else:
        lines.append("*No confusing-pair specimens were evaluated.*")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
