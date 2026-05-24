"""Ground-truth vision-only trait oracle loader.

Loads species trait profiles from species_traits.xml, filters to vision-only
traits, and produces two output formats:

1.  A2 flat dict  – human-readable keys appended to the LLM user message.
2.  B2 extractor output – dict shaped exactly like the trait extractor's
   ``visible_traits``, consumed by tree traversal, database comparison, and
   final LLM synthesis.

Both formats include a manually-curated ``stem_ring`` mapping because the XML
has no dedicated ring trait (ring presence is embedded in ``STEM/surface`` text).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data.dataset_utils import load_species_traits_xml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manual stem-ring mapping (XML has no dedicated STEM/ring trait)
# ---------------------------------------------------------------------------

_SPECIES_STEM_RING: Dict[str, str] = {
    "AG.AU": "absent",
    "AM.MU": "present",
    "AM.VI": "present",
    "BO.BA": "absent",
    "BO.ED": "absent",
    "CA.CI": "absent",
    "HY.PS": "absent",
    "CR.CO": "absent",
    "SP.CR": "absent",
    "CO.CO": "absent",
    "GY.ES": "absent",
    "GA.MA": "present",
    "PL.OS": "absent",
    "CA.TU": "absent",
    "HY.RE": "absent",
    "RU.IN": "absent",
    "RU.BA": "absent",
    "CAL.GI": "absent",
    "LY.PE": "absent",
    "RA.BO": "absent",
    "RA.PA": "absent",
    "LA.HE": "absent",
    "FO.BE": "absent",
}


# ---------------------------------------------------------------------------
# XML → extractor key mapping
# ---------------------------------------------------------------------------

# Maps (XML category, XML trait name) → extractor output key(s).
# All keys use American spelling to match the actual extractor code.
_XML_TO_EXTRACTOR_KEY: Dict[Tuple[str, str], str] = {
    ("CAP", "shape"): "cap_shape",
    ("CAP", "color"): "cap_color",
    ("CAP", "surface_texture"): "cap_surface",
    ("CAP", "size_cm"): "cap_size_cm",
    ("CAP", "margin"): "cap_margin",
    ("GILLS", "attachment"): "hymenophore_type",
    ("GILLS", "color"): "underside_color",
    ("GILLS", "density"): "gill_density",
    ("GILLS", "edge"): "gill_edge",
    ("STEM", "shape"): "stem_shape",
    ("STEM", "color"): "stem_color",
    ("STEM", "surface"): "stem_surface",
    ("STEM", "size_cm"): "stem_size_cm",
    ("FLESH", "color"): "flesh_color",
}

# Traits that are vision-only and should be included in oracle outputs.
_VISION_XML_TRAITS: set = set(_XML_TO_EXTRACTOR_KEY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_or_value(value: str) -> str:
    """Take first alternative from an OR-value like 'convex|flat'."""
    return value.split("|")[0].strip()


def _infer_hymenophore_type(gill_attachment: str) -> str:
    """Map XML GILLS/attachment value to extractor hymenophore_type."""
    val = gill_attachment.lower()
    if "pore" in val:
        return "pores"
    if "ridge" in val or "fold" in val:
        return "ridges"
    if "tooth" in val or "spine" in val:
        return "teeth"
    # Default for most gilled mushrooms
    return "gills"


def _infer_morphology_case(cap_shape: str, case_hint: str = "classical") -> str:
    """Derive morphology_case from cap_shape, matching derive_morphology_case()."""
    if case_hint == "puffball":
        return "puffball"
    if case_hint == "coral":
        return "coral"
    shape = cap_shape.lower()
    if "funnel" in shape or "depressed" in shape:
        return "classical_concave"
    if "convex" in shape or "flat" in shape or "bell" in shape:
        return "classical_convex"
    return "classical_unknown"


def _estimate_colour_ratios(dominant_color: str) -> Dict[str, float]:
    """Estimate colour_ratios dict from dominant color name.

    The real extractor computes pixel-level ratios. The oracle approximates
    them so downstream consumers that read colour_ratios don't crash.
    """
    val = dominant_color.lower()
    ratios = {
        "red": 0.0,
        "orange_red": 0.0,
        "orange_yellow": 0.0,
        "brown": 0.0,
        "white": 0.0,
        "dark": 0.0,
    }
    if "red" in val:
        ratios["red"] = 0.6
    if "orange" in val:
        ratios["orange_yellow"] = 0.5
    if "yellow" in val:
        ratios["orange_yellow"] = 0.5
    if "brown" in val:
        ratios["brown"] = 0.5
    if "white" in val or "cream" in val or "pale" in val:
        ratios["white"] = 0.5
    if "black" in val or "dark" in val or "grey" in val or "gray" in val:
        ratios["dark"] = 0.5
    # Ensure at least one channel is non-zero
    if sum(ratios.values()) == 0:
        ratios["white"] = 0.3
    return ratios


def _estimate_brightness(dominant_color: str) -> str:
    """Estimate brightness category from dominant color."""
    val = dominant_color.lower()
    if "white" in val or "cream" in val or "pale" in val or "yellow" in val:
        return "high"
    if "black" in val or "dark" in val:
        return "low"
    return "medium"


def _coarse_case(morphology_case: str) -> str:
    if morphology_case == "coral":
        return "coral"
    if morphology_case == "puffball":
        return "puffball"
    if morphology_case.startswith("classical"):
        return "classical"
    return "uncertain"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpeciesTraitOracle:
    """Loads ground-truth vision-only trait dicts from species_traits.xml."""

    def __init__(self, traits_xml_path: str):
        self.traits_df: pd.DataFrame = load_species_traits_xml(Path(traits_xml_path))
        self._by_species: Dict[str, pd.DataFrame] = {
            sid: df for sid, df in self.traits_df.groupby("species_id")
        }

    # -------------------------------------------------------------------
    # Internal: raw XML lookup
    # -------------------------------------------------------------------

    def _get_xml_traits(self, species_id: str) -> Dict[Tuple[str, str], str]:
        """Return vision-only traits as {(category, name): value} for a species."""
        df = self._by_species.get(species_id)
        if df is None:
            logger.warning("Species %s not found in species_traits.xml", species_id)
            return {}

        result: Dict[Tuple[str, str], str] = {}
        for _, row in df.iterrows():
            key = (row["trait_category"], row["trait_name"])
            if key in _VISION_XML_TRAITS:
                result[key] = row["trait_value"]
        return result

    # -------------------------------------------------------------------
    # A2: flat dict for LLM user message
    # -------------------------------------------------------------------

    def get_flat_dict(self, species_id: str) -> Dict[str, str]:
        """Return human-readable flat trait dict for A2.

        Example for AM.MU::

            {
                "cap_shape": "convex",
                "cap_color": "red with white spots",
                "cap_surface": "smooth",
                "gill_attachment": "free",
                "gill_color": "white",
                "stem_shape": "cylindrical",
                "stem_color": "white",
                "stem_ring": "present",
                "flesh_color": "white",
            }
        """
        xml_traits = self._get_xml_traits(species_id)
        flat: Dict[str, str] = {}

        # Map XML traits to human-readable keys
        mapping = {
            ("CAP", "shape"): "cap_shape",
            ("CAP", "color"): "cap_color",
            ("CAP", "surface_texture"): "cap_surface",
            ("CAP", "size_cm"): "cap_size_cm",
            ("CAP", "margin"): "cap_margin",
            ("GILLS", "attachment"): "gill_attachment",
            ("GILLS", "color"): "gill_color",
            ("GILLS", "density"): "gill_density",
            ("GILLS", "edge"): "gill_edge",
            ("STEM", "shape"): "stem_shape",
            ("STEM", "color"): "stem_color",
            ("STEM", "surface"): "stem_surface",
            ("STEM", "size_cm"): "stem_size_cm",
            ("FLESH", "color"): "flesh_color",
        }

        for xml_key, flat_key in mapping.items():
            if xml_key in xml_traits:
                flat[flat_key] = xml_traits[xml_key]

        # Add manual stem_ring
        flat["stem_ring"] = _SPECIES_STEM_RING.get(species_id, "unknown")

        return flat

    # -------------------------------------------------------------------
    # B2: extractor-shaped output
    # -------------------------------------------------------------------

    def get_extractor_output(
        self, species_id: str, case: str = "classical"
    ) -> Dict[str, Any]:
        """Return a dict shaped like the trait extractor's ``visible_traits``.

        This is the signal fed to tree traversal, database comparison, and
        final LLM synthesis in B2.

        Args:
            species_id: Target species ID.
            case: Morphology case hint — one of ``classical``, ``puffball``,
                ``coral``, ``uncertain``.
        """
        xml_traits = self._get_xml_traits(species_id)

        # Resolve OR-values to deterministic single values
        resolved: Dict[Tuple[str, str], str] = {
            k: _resolve_or_value(v) for k, v in xml_traits.items()
        }

        # Core traits from XML mapping
        cap_shape = resolved.get(("CAP", "shape"), "unknown")
        cap_color = resolved.get(("CAP", "color"), "unknown")
        cap_surface = resolved.get(("CAP", "surface_texture"), "unknown")
        cap_size = resolved.get(("CAP", "size_cm"), "unknown")
        cap_margin = resolved.get(("CAP", "margin"), "unknown")
        gill_attachment = resolved.get(("GILLS", "attachment"), "unknown")
        gill_color = resolved.get(("GILLS", "color"), "unknown")
        gill_density = resolved.get(("GILLS", "density"), "unknown")
        gill_edge = resolved.get(("GILLS", "edge"), "unknown")
        stem_shape = resolved.get(("STEM", "shape"), "unknown")
        stem_color = resolved.get(("STEM", "color"), "unknown")
        stem_surface = resolved.get(("STEM", "surface"), "unknown")
        stem_size = resolved.get(("STEM", "size_cm"), "unknown")
        flesh_color = resolved.get(("FLESH", "color"), "unknown")

        # Derived traits
        hymenophore_type = _infer_hymenophore_type(gill_attachment)
        morphology_case = _infer_morphology_case(cap_shape, case)
        coarse = _coarse_case(morphology_case)
        stem_ring = _SPECIES_STEM_RING.get(species_id, "unknown")
        colour_ratios = _estimate_colour_ratios(cap_color)
        brightness = _estimate_brightness(cap_color)
        has_ridges = hymenophore_type == "ridges"

        # Case-specific keys and detected_parts
        detected_parts: List[str]
        if case == "puffball":
            detected_parts = ["whole"]
        elif case == "coral":
            detected_parts = ["coral"]
        else:
            detected_parts = ["cap", "stem", "underside"]

        # Build the extractor-shaped visible_traits dict
        visible_traits: Dict[str, Any] = {
            # Legacy compatibility keys
            "dominant_color": cap_color,
            "secondary_color": cap_color,  # XML has no secondary; duplicate dominant
            "cap_shape": cap_shape,
            "surface_texture": cap_surface,
            "has_ridges": has_ridges,
            "brightness": brightness,
            "colour_ratios": colour_ratios,

            # Part-aware keys
            "morphology_case": morphology_case,
            "coarse_case": coarse,
            "detected_parts": detected_parts,
            "hymenophore_type": hymenophore_type,
            "hymenophore_confidence": 0.95,
            "cap_color": cap_color,
            "stem_color": stem_color,
            "underside_color": gill_color,
            "whole_color": cap_color,
            "cap_surface": cap_surface,
            "stem_ring": stem_ring,
            "stem_surface": stem_surface,
            "clustered_growth": False,
            "mask_used": True,

            # Confidence metadata (uniform high confidence for oracle)
            "trait_confidence": {
                "cap_color": 0.95,
                "cap_shape": 0.95,
                "cap_surface": 0.95,
                "stem_color": 0.95,
                "stem_surface": 0.95,
                "stem_ring": 0.95,
                "underside_color": 0.95,
                "hymenophore_type": 0.95,
                "flesh_color": 0.95,
            },
            "trait_source_by_key": {
                "cap_color": "oracle",
                "cap_shape": "oracle",
                "cap_surface": "oracle",
                "stem_color": "oracle",
                "stem_surface": "oracle",
                "stem_ring": "oracle",
                "underside_color": "oracle",
                "hymenophore_type": "oracle",
                "flesh_color": "oracle",
            },
        }

        # Case-specific optional keys
        if case == "puffball":
            visible_traits["puffball_surface"] = cap_surface
            visible_traits["puffball_roundness"] = 0.85  # approximate high roundness


        # Size keys (optional, present when available)
        if cap_size != "unknown":
            visible_traits["cap_size_cm"] = cap_size
        if stem_size != "unknown":
            visible_traits["stem_size_cm"] = stem_size
        if cap_margin != "unknown":
            visible_traits["cap_margin"] = cap_margin
        if gill_density != "unknown":
            visible_traits["gill_density"] = gill_density
        if gill_edge != "unknown":
            visible_traits["gill_edge"] = gill_edge
        if flesh_color != "unknown":
            visible_traits["flesh_color"] = flesh_color

        return visible_traits

    # -------------------------------------------------------------------
    # B2 fallback: species trait dict with OR-values preserved
    # -------------------------------------------------------------------

    def get_species_trait_dict(self, species_id: str) -> Dict[str, str]:
        """Return flat vision-only trait dict with OR-values preserved.

        This is used when the tree gets stuck and needs to see the full
        variability (e.g., ``convex|flat``) to infer the correct answer.
        """
        xml_traits = self._get_xml_traits(species_id)
        result: Dict[str, str] = {}
        for xml_key, extractor_key in _XML_TO_EXTRACTOR_KEY.items():
            if xml_key in xml_traits:
                result[extractor_key] = xml_traits[xml_key]
        result["stem_ring"] = _SPECIES_STEM_RING.get(species_id, "unknown")
        return result
