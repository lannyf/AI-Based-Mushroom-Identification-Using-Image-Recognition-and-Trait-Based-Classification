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


_SYSTEM_A_METHODS = {"cnn", "tree", "db", "llm"}
_SYSTEM_B_METHODS = {"unified"}


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
        "agreement",
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
                "agreement": entry["agreement"],
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
    """Write a thesis-ready Markdown report separating System A and System B."""
    lines: List[str] = ["# Comparative Benchmark Results\n"]

    per_method = metrics.get("per_method", {})

    # --- 1. System A — Standalone Methods --------------------------------
    lines.append("## 1. System A — Standalone Methods\n")
    lines.append("*Each method operates independently without access to the others.*\n")
    lines.append("| Method | Accuracy | Coverage | Mean Time (ms) |")
    lines.append("|--------|----------|----------|----------------|")
    for method, m in per_method.items():
        if method in _SYSTEM_A_METHODS:
            lines.append(
                f"| {method} | {m['accuracy']:.1%} | {m['coverage']:.1%} | {m['mean_time_ms']:.1f} |"
            )
    lines.append("")

    # --- 2. System B — Unified LLM Synthesis -----------------------------
    lines.append("## 2. System B — Unified LLM Synthesis\n")
    lines.append("*The LLM aggregates signals from all System A subsystems into a single prediction.*\n")
    lines.append("| Method | Accuracy | Coverage | Mean Time (ms) |")
    lines.append("|--------|----------|----------|----------------|")
    for method, m in per_method.items():
        if method in _SYSTEM_B_METHODS:
            lines.append(
                f"| {method} | {m['accuracy']:.1%} | {m['coverage']:.1%} | {m['mean_time_ms']:.1f} |"
            )
    lines.append("")

    # --- 3. Raw Accuracy Difference: System B vs Each System A Method ---
    lines.append("## 3. Raw Accuracy Difference: System B vs System A\n")
    lines.append("*Simple accuracy difference between Unified and each standalone method.*\n")
    lines.append("| Comparison | Unified Acc | Standalone Acc | Difference |")
    lines.append("|------------|-------------|----------------|------------|")
    unified_m = per_method.get("unified", {})
    unified_acc = unified_m.get("accuracy", 0)
    for method, m in per_method.items():
        if method in _SYSTEM_A_METHODS:
            diff = unified_acc - m["accuracy"]
            sign = "+" if diff >= 0 else ""
            lines.append(
                f"| unified vs {method} | {unified_acc:.1%} | {m['accuracy']:.1%} | {sign}{diff:.1%} |"
            )
    lines.append("")

    # --- 4. Confusing-Pair Breakdown ------------------------------------
    lines.append("## 4. Confusing-Pair Breakdown\n")
    confusing = metrics.get("confusing_pairs", {})
    if confusing:
        lines.append("| Pair | N | CNN | Tree | DB | LLM | Unified | Agr |")
        lines.append("|------|---|-----|------|----|-----|---------|-----|")
        for pair_name, p in confusing.items():
            lines.append(
                f"| {pair_name} | {p['n']} | {p['cnn_acc']:.0%} | {p['tree_acc']:.0%} | "
                f"{p['db_acc']:.0%} | {p.get('llm_acc', 0):.0%} | {p['unified_acc']:.0%} | {p['agreement_rate']:.0%} |"
            )
    else:
        lines.append("*No confusing-pair specimens were evaluated.*")
    lines.append("")

    # --- 5. Cases Where System B Outperformed All System A ---------------
    lines.append("## 5. Cases Where System B Outperformed All System A Methods\n")
    wins = [e for e in per_specimen if e.get("unified_outperforms_all", False)]
    if wins:
        lines.append("| Specimen | GT | CNN | Tree | DB | LLM | Unified | Reasoning |")
        lines.append("|----------|----|-----|------|----|-----|---------|-----------|")
        for e in wins:
            r = e["results"]
            lines.append(
                f"| {e['specimen_id']} | {e['species_id']} | "
                f"{r['cnn']['top_species']} | {r['tree']['top_species']} | "
                f"{r['db']['top_species']} | {r['llm']['top_species']} | {r['unified']['top_species']} | "
                f"{r['unified'].get('reasoning', '')[:60]}… |"
            )
    else:
        lines.append("*No such cases found.*")
    lines.append("")

    # --- 6. Cases Where System B Was Wrong But System A Was Right --------
    lines.append("## 6. Cases Where System B Was Wrong But a System A Method Was Right\n")
    losses = [e for e in per_specimen if e.get("unified_wrong_but_standalone_right", False)]
    if losses:
        lines.append("| Specimen | GT | CNN | Tree | DB | LLM | Unified | Notes |")
        lines.append("|----------|----|-----|------|----|-----|---------|-------|")
        for e in losses:
            r = e["results"]
            lines.append(
                f"| {e['specimen_id']} | {e['species_id']} | "
                f"{r['cnn']['top_species']} | {r['tree']['top_species']} | "
                f"{r['db']['top_species']} | {r['llm']['top_species']} | {r['unified']['top_species']} | {e.get('notes', '')} |"
            )
    else:
        lines.append("*No such cases found.*")
    lines.append("")

    # --- 7. Agreement Statistics ----------------------------------------
    lines.append("## 7. Agreement Statistics\n")
    lines.append("| Agreement Level | Count | % | Avg System B Accuracy |")
    lines.append("|-----------------|-------|---|-----------------------|")
    for level, s in metrics.get("agreement_stats", {}).items():
        lines.append(
            f"| {level} | {s['count']} | {s['pct']:.1%} | {s['unified_accuracy']:.1%} |"
        )
    lines.append("")

    # --- 8. Scenario Breakdown ------------------------------------------
    lines.append("## 8. Accuracy by Scenario\n")
    lines.append("| Scenario | N | CNN | Tree | DB | LLM | Unified |")
    lines.append("|----------|---|-----|------|----|-----|---------|")
    for scenario, s in metrics.get("by_scenario", {}).items():
        lines.append(
            f"| {scenario} | {s['n']} | {s['cnn_acc']:.0%} | {s['tree_acc']:.0%} | "
            f"{s['db_acc']:.0%} | {s.get('llm_acc', 0):.0%} | {s['unified_acc']:.0%} |"
        )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
