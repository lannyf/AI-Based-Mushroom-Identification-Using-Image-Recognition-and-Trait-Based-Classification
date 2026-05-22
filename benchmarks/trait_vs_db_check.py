#!/usr/bin/env python3
"""
Quick trait-extraction vs trait-database sanity check.

Selects 10 diverse species from the benchmark manifest, runs the unified
pipeline (or standalone extractor) on one specimen per species, and compares
the extracted visible traits against the ground-truth entries in
species_traits.xml.

Usage:
    python benchmarks/trait_vs_db_check.py
"""

from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Set

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Remove the script's directory from path so benchmarks/config.py doesn't shadow config/
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.path.insert(0, str(PROJECT_ROOT))
MANIFEST_CSV = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"
TRAITS_XML = PROJECT_ROOT / "data" / "raw" / "species_traits.xml"


def load_species_traits(xml_path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse species_traits.xml into a flat dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    db: Dict[str, Dict[str, Any]] = {}
    for species in root.findall("species"):
        sid = species.get("id", "")
        traits: Dict[str, Any] = {}
        for group in species.findall("trait_group"):
            category = group.get("category", "").lower()
            for trait in group.findall("trait"):
                name = trait.get("name", "")
                value = (trait.text or "").strip()
                key = f"{category}_{name}"
                traits[key] = value
        db[sid] = traits
    return db


def parse_trait_value(value: str) -> Set[str]:
    """Split pipe-separated trait values into a set of normalized strings."""
    return set(v.strip().lower() for v in value.split("|") if v.strip())


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick_specimen_for_species(manifest: List[Dict[str, str]], species_id: str) -> Dict[str, str] | None:
    for row in manifest:
        if row["species_id"] == species_id:
            return row
    return None


def resolve_image_path(row: Dict[str, str]) -> tuple[Path | None, Path | None]:
    """Return absolute paths for above/below images."""
    above = row.get("above_image_path", "")
    below = row.get("below_image_path", "")
    # Paths in manifest are relative to the manifest file location
    manifest_dir = MANIFEST_CSV.parent
    a_path = (manifest_dir / above).resolve() if above else None
    b_path = (manifest_dir / below).resolve() if below else None
    return a_path, b_path


def run_pipeline(above_path: Path, below_path: Path) -> Dict[str, Any]:
    from models.unified_pipeline import UnifiedPipeline
    from models.mushroom_segmenter import get_segmenter
    pipeline = UnifiedPipeline(segmenter=get_segmenter(), auto_init_llm=False)
    above_bytes = above_path.read_bytes()
    below_bytes = below_path.read_bytes() if below_path.exists() else b""
    return pipeline.run(above_bytes, below_bytes or above_bytes)


def match_color(extracted: str, db_values: str) -> bool:
    """Check if extracted colour matches any of the DB colour options."""
    ext = extracted.lower().strip()
    db_set = parse_trait_value(db_values)
    # Direct match
    if ext in db_set:
        return True
    # Partial overlap for composite names
    for db_val in db_set:
        if ext in db_val or db_val in ext:
            return True
    return False


def match_shape(extracted: str, db_values: str) -> bool:
    """Check if extracted shape matches any of the DB shape options."""
    ext = extracted.lower().strip()
    db_set = parse_trait_value(db_values)
    return ext in db_set


def main():
    import sys
    # Allow overriding part-aware flag via env var for quick A/B test
    part_aware = False

    print("=" * 70)
    print("Trait Extraction vs Trait Database — 10-species sanity check")
    print(f"Part-aware traits: {part_aware}")
    print("=" * 70)

    db = load_species_traits(TRAITS_XML)
    manifest = load_manifest(MANIFEST_CSV)

    # 10 diverse species covering classical, bolete, chanterelle, coral, puffball
    selected_species = [
        "AM.MU",   # Fly agaric (classical, red cap)
        "BO.ED",   # Porcini (bolete, brown cap, pores)
        "CA.CI",   # Chanterelle (yellow, ridges)
        "CR.CO",   # Black trumpet (funnel, black)
        "HY.PS",   # False chanterelle (orange, gills)
        "LA.HE",   # Shaggy ink cap (classical, white, deliquescing)
        "RA.BO",   # Crown coral (coral)
        "SP.CR",   # Common puffball (puffball)
        "AG.AU",   # Field mushroom (classical, white)
        "PL.OS",   # Oyster mushroom (classical, white/grey, gills)
    ]

    cap_color_hits = 0
    cap_shape_hits = 0
    stem_color_hits = 0
    hymenophore_hits = 0
    total_evaluated = 0

    for species_id in selected_species:
        row = pick_specimen_for_species(manifest, species_id)
        if row is None:
            print(f"\n{species_id}: NOT FOUND in manifest — skipping")
            continue

        above_path, below_path = resolve_image_path(row)
        if above_path is None or not above_path.exists():
            print(f"\n{species_id}: image missing — skipping")
            continue

        print(f"\n--- {species_id} ({row.get('specimen_id','')}) ---")
        result = run_pipeline(above_path, below_path or above_path)
        traits = result["traits"]["merged"]
        case = result["case"]["case"]
        detected_parts = result["case"].get("detected_parts", [])

        print(f"  Pipeline case: {case}")
        print(f"  Detected parts: {detected_parts}")

        db_traits = db.get(species_id, {})
        if not db_traits:
            print(f"  WARNING: no DB traits for {species_id}")
            continue

        total_evaluated += 1

        # Cap color
        extracted_cap_color = traits.get("cap_color") or traits.get("dominant_color") or "unknown"
        db_cap_color = db_traits.get("cap_color", "")
        cap_color_match = match_color(extracted_cap_color, db_cap_color) if db_cap_color else False
        if cap_color_match:
            cap_color_hits += 1
        print(f"  Cap color: extracted='{extracted_cap_color}'  db='{db_cap_color}'  -> {'✓' if cap_color_match else '✗'}")

        # Cap shape
        extracted_cap_shape = traits.get("cap_shape", "unknown")
        db_cap_shape = db_traits.get("cap_shape", "")
        cap_shape_match = match_shape(extracted_cap_shape, db_cap_shape) if db_cap_shape else False
        if cap_shape_match:
            cap_shape_hits += 1
        print(f"  Cap shape: extracted='{extracted_cap_shape}'  db='{db_cap_shape}'  -> {'✓' if cap_shape_match else '✗'}")

        # Stem color (only for classical / bolete)
        if case in ("classical", "puffball", "uncertain"):
            extracted_stem_color = traits.get("stem_color", "unknown")
            db_stem_color = db_traits.get("stem_color", "")
            stem_match = match_color(extracted_stem_color, db_stem_color) if db_stem_color else False
            if stem_match:
                stem_color_hits += 1
            print(f"  Stem color: extracted='{extracted_stem_color}'  db='{db_stem_color}'  -> {'✓' if stem_match else '✗'}")

        # Hymenophore type (only when underside is available)
        if "underside" in detected_parts or case in ("classical", "uncertain"):
            extracted_hym = traits.get("hymenophore_type", "unknown")
            # DB stores this under gills_attachment or gills_color usually; map roughly
            db_gills_attach = db_traits.get("gills_attachment", "")
            db_gills_color = db_traits.get("gills_color", "")
            hym_match = False
            if "pores" in db_gills_attach.lower() and extracted_hym == "pores":
                hym_match = True
            elif "gills" in db_gills_attach.lower() and extracted_hym == "gills":
                hym_match = True
            elif "ridges" in db_gills_attach.lower() and extracted_hym == "ridges":
                hym_match = True
            if hym_match:
                hymenophore_hits += 1
            print(f"  Hymenophore: extracted='{extracted_hym}'  db_attach='{db_gills_attach}'  -> {'✓' if hym_match else '✗'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Species evaluated: {total_evaluated}")
    if total_evaluated > 0:
        print(f"Cap color:      {cap_color_hits}/{total_evaluated} correct ({cap_color_hits/total_evaluated:.0%})")
        print(f"Cap shape:      {cap_shape_hits}/{total_evaluated} correct ({cap_shape_hits/total_evaluated:.0%})")
        print(f"Stem color:     {stem_color_hits}/{total_evaluated} correct ({stem_color_hits/total_evaluated:.0%})")
        print(f"Hymenophore:    {hymenophore_hits}/{total_evaluated} correct ({hymenophore_hits/total_evaluated:.0%})")


if __name__ == "__main__":
    main()
