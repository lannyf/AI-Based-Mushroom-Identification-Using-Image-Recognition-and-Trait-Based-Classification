"""
Unified Mushroom Identification Pipeline

Orchestrates the full identification flow:
  1. YOLO segmentation (above + below photos)
  2. Case detection (classical / coral / puffball / uncertain)
  3. Trait extraction per photo (masked where possible)
  4. CNN prediction with uncertainty flags
  5. Key-tree auto-traversal
  6. Trait-database comparison
  7. LLM synthesis of all signals
  8. Agreement evaluation

The pipeline is stateless — each call is independent.  It is designed to be
invoked from the FastAPI ``/identify/unified`` endpoint.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from models.cnn_classifier import get_classifier
from models.key_tree_parser import KeyTreeParser
from models.key_tree_traversal import KeyTreeEngine
from models.llm_classifier import LLMClassifier, UnifiedPredictionResult
from models.tree_path_validator import TreePathValidator
from models.mushroom_segmenter import Segmenter
from models.trait_database_comparator import TraitDatabaseComparator
from models.visual_trait_extractor import extract as _extract_traits
from models.yolo_part_masks import build_part_masks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_XML_PATH = PROJECT_ROOT / "data" / "raw" / "key.xml"
SPECIES_CSV = PROJECT_ROOT / "data" / "raw" / "species.csv"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(image_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _image_to_llm_b64(image_bytes: bytes, max_side: int = 256, quality: int = 80) -> str:
    """Downsample an image before sending it to the local vision LLM."""
    import base64

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    pil.thumbnail((max_side, max_side), resampling)

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


PHOTO_PREFERENCE = {
    "cap_color": "above",
    "cap_shape": "above",
    "cap_surface": "above",
    "underside_color": "below",
    "hymenophore_type": "below",
    "stem_color": "below",
    "stem_ring": "below",
    "stem_surface": "below",
    "puffball_surface": "above",
}


def _merge_traits(
    above_traits: Dict[str, Any],
    below_traits: Dict[str, Any],
    case: str,
) -> Dict[str, Any]:
    """
    Merge trait dicts from above and below photos, preferring the most
    informative value for each key.
    """
    merged: Dict[str, Any] = {}
    all_keys = set(above_traits.keys()) | set(below_traits.keys())

    for key in all_keys:
        a = above_traits.get(key)
        b = below_traits.get(key)
        if key == "colour_ratios":
            if a and b:
                all_ratio_keys = set(a.keys()) | set(b.keys())
                merged[key] = {
                    k: round((a.get(k, 0.0) + b.get(k, 0.0)) / 2, 3)
                    for k in all_ratio_keys
                }
            elif a:
                merged[key] = dict(a)
            elif b:
                merged[key] = dict(b)
        elif key in ("dominant_color", "secondary_color", "cap_shape",
                     "surface_texture", "brightness"):
            if case == "classical":
                if key in ("dominant_color", "cap_shape", "surface_texture"):
                    merged[key] = a if a and a != "unknown" else (b if b and b != "unknown" else (a or b))
                else:
                    merged[key] = a if a and a != "unknown" else (b if b and b != "unknown" else (a or b))
            else:
                merged[key] = a if a and a != "unknown" else (b if b and b != "unknown" else (a or b))
        elif key == "has_ridges":
            # Below photo often better for underside structures; use OR logic
            merged[key] = bool(a) if a is not None else bool(b) if b is not None else False
        elif key in PHOTO_PREFERENCE:
            pref = PHOTO_PREFERENCE[key]
            preferred = a if pref == "above" else b
            fallback = b if pref == "above" else a
            if preferred is not None and preferred != "unknown":
                merged[key] = preferred
            else:
                merged[key] = fallback
        elif key in ("trait_confidence", "trait_source_by_key"):
            # Merge dicts: above wins on overlap unless below has higher confidence
            merged_a = dict(a) if a else {}
            merged_b = dict(b) if b else {}
            result = dict(merged_a)
            for k, v in merged_b.items():
                if k not in result:
                    result[k] = v
            merged[key] = result
        else:
            merged[key] = a if a is not None else b

    # Add provenance
    merged["photo_count"] = 2
    merged["case"] = case
    return merged


def _evaluate_agreement(
    cnn: Dict[str, Any],
    tree: Dict[str, Any],
    db: Dict[str, Any],
    llm: UnifiedPredictionResult,
) -> str:
    """
    Evaluate agreement between all four prediction sources.
    Returns: "agree" | "disagree" | "partial" | "inconclusive"
    """
    signals: Dict[str, Optional[str]] = {
        "cnn": cnn.get("species"),
        "tree": tree.get("species") if tree.get("status") == "conclusion" else None,
        "db": db.get("candidate", {}).get("english_name") if db.get("status") == "ok" else None,
        "llm": llm.top_species if llm.top_species not in ("Unknown", "Error") else None,
    }

    # Filter out None / uncertain signals
    valid = {k: v for k, v in signals.items() if v is not None}
    if len(valid) < 2:
        return "inconclusive"

    values = list(valid.values())
    unique = set(v.lower() for v in values)

    if len(unique) == 1:
        return "agree"

    # Check if at least 3 agree (or 2 out of 2)
    from collections import Counter
    counts = Counter(v.lower() for v in values)
    most_common = counts.most_common(1)[0]
    if most_common[1] >= 3 or (len(values) == 2 and most_common[1] == 2):
        return "partial" if len(unique) > 1 else "agree"

    if most_common[1] == 1:
        return "disagree"

    return "partial"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class UnifiedPipeline:
    """
    End-to-end unified identification pipeline.

    Usage
    -----
    pipeline = UnifiedPipeline()
    result = pipeline.run(above_bytes, below_bytes)
    """

    def __init__(
        self,
        segmenter: Optional[Segmenter] = None,
        key_xml_path: str = str(KEY_XML_PATH),
        data_raw_dir: str = str(DATA_RAW_DIR),
        llm_backend: Optional[LLMClassifier] = None,
        auto_init_llm: bool = True,
    ):
        self.segmenter = segmenter
        self.key_xml_path = key_xml_path
        self.key_tree = KeyTreeEngine(key_xml_path)
        self.comparator = TraitDatabaseComparator(data_raw_dir)
        self.llm = llm_backend

        # Parse key.xml for LLM injection
        try:
            self.key_tree_parser = KeyTreeParser(key_xml_path)
            self.key_tree_text = self.key_tree_parser.get_prompt_injection()
        except Exception as exc:
            logger.warning("Could not parse key.xml for LLM injection: %s", exc)
            self.key_tree_text = ""

        # Lazy-init LLM if not provided but Ollama is available
        if self.llm is None and auto_init_llm:
            from models.llm_classifier import OllamaBackend
            if OllamaBackend.is_available():
                try:
                    self.llm = LLMClassifier(
                        backend_type="ollama",
                        key_tree_text=self.key_tree_text,
                    )
                    logger.info("UnifiedPipeline: LLM auto-initialised")
                except Exception as exc:
                    logger.warning("UnifiedPipeline: LLM init failed: %s", exc)

    # ------------------------------------------------------------------

    def run(
        self,
        above_image_bytes: bytes,
        below_image_bytes: bytes,
        species_id: Optional[str] = None,
        oracle_trait_provider = None,
    ) -> Dict[str, Any]:
        """
        Run the full unified pipeline on an above-photo and a below-photo.

        Args:
            above_image_bytes: Cap/top photo.
            below_image_bytes: Underside/gills photo.
            species_id: Target species ID (required when oracle_trait_provider is set).
            oracle_trait_provider: Optional SpeciesTraitOracle for B2 mode.

        Returns a dict with:
          {
            "case": {...},
            "segmentation": {"above": ..., "below": ...},
            "traits": {"above": ..., "below": ..., "merged": ...},
            "cnn": {...},
            "tree": {...},
            "database": {...},
            "llm": {...},
            "agreement": "agree|disagree|partial|inconclusive",
            "final_recommendation": {...},
            "processing_time_ms": float,
          }
        """
        t0 = time.time()

        # ---- 1. Segmentation ------------------------------------------------
        above_seg = self._segment(above_image_bytes)
        below_seg = self._segment(below_image_bytes)

        # ---- 2. Case detection & 3. Trait extraction ------------------------
        from models.yolo_part_masks import build_part_masks
        from models.mushroom_segmenter import detect_case_from_masks

        pil = Image.open(io.BytesIO(above_image_bytes)).convert("RGB")
        H, W = np.array(pil).shape[:2]
        above_masks = build_part_masks(above_seg.get("instances", []), (H, W))

        pil = Image.open(io.BytesIO(below_image_bytes)).convert("RGB")
        H, W = np.array(pil).shape[:2]
        below_masks = build_part_masks(below_seg.get("instances", []), (H, W))

        case = detect_case_from_masks(above_masks, below_masks)

        above_traits = _extract_traits(above_image_bytes, part_masks=above_masks)["visible_traits"]
        below_traits = _extract_traits(below_image_bytes, part_masks=below_masks)["visible_traits"]

        merged_traits = _merge_traits(above_traits, below_traits, case["case"])

        # Pipeline owns morphology_case confidence (Codex guardrail 8.6)
        if "trait_confidence" not in merged_traits:
            merged_traits["trait_confidence"] = {}
        merged_traits["trait_confidence"]["morphology_case"] = case.get("confidence", 0.0)
        if "trait_source_by_key" not in merged_traits:
            merged_traits["trait_source_by_key"] = {}
        merged_traits["trait_source_by_key"]["morphology_case"] = "pipeline_case_detection"
        # Ensure pipeline case is authoritative (overrides extractor-derived case)
        merged_traits["morphology_case"] = case["case"]
        merged_traits["coarse_case"] = case["case"]

        # ---- B2: swap in oracle traits if provider is given -----------------
        traits_for_tools = merged_traits
        oracle_used = False
        if oracle_trait_provider is not None and species_id:
            oracle_visible_traits = oracle_trait_provider.get_extractor_output(
                species_id, case=case["case"]
            )
            # Preserve pipeline case metadata so downstream logic is consistent
            oracle_visible_traits["morphology_case"] = case["case"]
            oracle_visible_traits["coarse_case"] = case["case"]
            traits_for_tools = oracle_visible_traits
            oracle_used = True

        # ---- 4. CNN prediction ----------------------------------------------
        # Run on the photo with the best detection
        best_photo_bytes = above_image_bytes
        above_best_conf = max(
            (i.get("model_confidence", 0.0) for i in above_seg.get("instances", [])),
            default=0.0,
        )
        below_best_conf = max(
            (i.get("model_confidence", 0.0) for i in below_seg.get("instances", [])),
            default=0.0,
        )
        if below_best_conf > above_best_conf:
            best_photo_bytes = below_image_bytes

        cnn = get_classifier()
        cnn_pred = cnn.predict_with_uncertainty(best_photo_bytes)

        # Encode images once for LLM (tree nav + synthesis)
        if os.environ.get("UNIFIED_LLM_NO_IMAGES", "").lower() in ("1", "true", "yes"):
            images_b64 = None
        else:
            images_b64 = [
                _image_to_llm_b64(above_image_bytes),
                _image_to_llm_b64(below_image_bytes),
            ]

        # ---- 5. Tree traversal (LLM navigation) -----------------------------
        # Use the LLM to navigate the polytomous key in one shot.
        # The LLM receives the tree structure + traits + CNN hint and
        # returns its chosen path. A validator checks it against key.xml.
        tree_res: Dict[str, Any]
        if self.llm is not None:
            nav_result = self.llm.navigate_tree(
                visible_traits=traits_for_tools,
                cnn_prediction=cnn_pred,
                case_info=case,
                images_b64=images_b64,
            )

            # Validate the LLM's path
            validator = TreePathValidator(self.key_xml_path)
            validation = validator.validate(nav_result.get("tree_path", []))

            if validation["valid"] and validation["decision"]:
                tree_res = {
                    "status": "conclusion",
                    "species": validation["decision"],
                    "edibility_label": validation.get("edibility", ""),
                    "path": [f"{q} → {a}" for q, a in validation.get("best_path", [])],
                    "auto_answered": [],
                    "llm_navigated": True,
                    "llm_conclusion": nav_result.get("conclusion"),
                    "llm_confidence": nav_result.get("confidence"),
                    "llm_reasoning": nav_result.get("reasoning"),
                }
            else:
                # Path invalid or incomplete — return partial context so the
                # LLM synthesis still sees how far the tree got.
                partial = validator.get_partial_context(nav_result.get("tree_path", []))
                tree_res = {
                    "status": "question",
                    "species": None,
                    "question": partial.get("question", ""),
                    "options": partial.get("options", []),
                    "path": [f"{s.get('question', '')} → {s.get('answer', '')}" for s in partial.get("partial_path", [])],
                    "auto_answered": [],
                    "llm_navigated": True,
                    "llm_conclusion": nav_result.get("conclusion"),
                    "llm_confidence": nav_result.get("confidence"),
                    "llm_reasoning": nav_result.get("reasoning"),
                    "partial_depth": partial.get("depth", 0),
                    "validation_error": "Path incomplete — returned deepest valid node",
                }
        else:
            # Fallback to programmatic engine if LLM unavailable
            ml_hint = None
            if cnn_pred.get("species"):
                ml_hint = {
                    "top_species": cnn_pred["species"],
                    "confidence": cnn_pred["confidence"],
                }
            tree_res = self.key_tree.start_session(
                session_id=None,
                visible_traits=traits_for_tools,
                ml_hint=ml_hint,
            )

        # If tree reached conclusion, run DB comparison

        # ---- 6. Database comparison -----------------------------------------
        db_res: Dict[str, Any] = {"status": "skipped", "reason": "No tree conclusion"}
        if tree_res.get("status") == "conclusion":
            db_res = self.comparator.compare(
                tree_res.get("species", ""),
                traits_for_tools,
            )
        elif cnn_pred.get("species"):
            # Fallback: compare CNN top species
            db_res = self.comparator.compare(
                cnn_pred["species"],
                traits_for_tools,
            )

        # ---- 7. LLM synthesis -----------------------------------------------
        # Images were already encoded in step 5 for reuse.
        llm_res = self._run_llm(
            traits_for_tools, cnn_pred, tree_res, db_res, case, images_b64
        )

        # ---- 8. Agreement evaluation ----------------------------------------
        agreement = _evaluate_agreement(cnn_pred, tree_res, db_res, llm_res)

        # Override LLM's own agreement with ours if it differs
        if llm_res.agreement_state != agreement:
            logger.debug(
                "LLM agreement (%s) vs computed (%s) — using computed",
                llm_res.agreement_state, agreement,
            )

        processing_time = (time.time() - t0) * 1000

        # ---- Build final recommendation -------------------------------------
        final_rec = self._build_final_recommendation(
            llm_res, cnn_pred, tree_res, db_res, agreement
        )

        # Sanitize segmentation output — remove raw numpy arrays (not JSON-serializable)
        sanitized_seg = {
            "above": self._sanitize_seg(above_seg),
            "below": self._sanitize_seg(below_seg),
        }

        return {
            "case": case,
            "segmentation": sanitized_seg,
            "traits": {
                "above": above_traits,
                "below": below_traits,
                "merged": merged_traits,
            },
            "cnn": cnn_pred,
            "tree": tree_res,
            "database": db_res,
            "llm": llm_res.to_dict(),
            "agreement": agreement,
            "final_recommendation": final_rec,
            "processing_time_ms": round(processing_time, 2),
            "oracle_used": oracle_used,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_seg(seg_result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove numpy arrays from segmentation result for JSON serialization."""
        out = {"instances": [], "selected_index": seg_result.get("selected_index")}
        if "error" in seg_result:
            out["error"] = seg_result["error"]
        for inst in seg_result.get("instances", []):
            clean = {k: v for k, v in inst.items() if not isinstance(v, np.ndarray)}
            out["instances"].append(clean)
        return out

    def _segment(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run segmentation, returning empty result on failure."""
        if self.segmenter is None:
            return {"instances": [], "selected_index": None}
        try:
            return self.segmenter.segment(image_bytes)
        except Exception as exc:
            logger.warning("Segmentation failed: %s", exc)
            return {"instances": [], "selected_index": None, "error": str(exc)}

    def _run_llm(
        self,
        merged_traits: Dict[str, Any],
        cnn_pred: Dict[str, Any],
        tree_res: Dict[str, Any],
        db_res: Dict[str, Any],
        case_info: Dict[str, Any],
        images_b64: Optional[List[str]] = None,
    ) -> UnifiedPredictionResult:
        """Run unified LLM classification if available; otherwise return a fallback."""
        if self.llm is None:
            return UnifiedPredictionResult(
                top_species="Unknown",
                top_confidence=0.0,
                predictions=[],
                reasoning="LLM not available.",
                safety_warnings=["LLM unavailable — identification based on subsystems only."],
                model_used="none",
                processing_time_ms=0.0,
                agreement_state="inconclusive",
            )
        try:
            return self.llm.unified_classify(
                visible_traits=merged_traits,
                cnn_prediction=cnn_pred,
                tree_result=tree_res,
                db_result=db_res,
                case_info=case_info,
                images_b64=images_b64,
            )
        except Exception as exc:
            logger.error("LLM unified classify failed: %s", exc)
            return UnifiedPredictionResult(
                top_species="Unknown",
                top_confidence=0.0,
                predictions=[],
                reasoning=f"LLM error: {exc}",
                safety_warnings=["LLM error — consult other signals."],
                model_used=self.llm.backend_type,
                processing_time_ms=0.0,
                agreement_state="inconclusive",
            )

    def _build_final_recommendation(
        self,
        llm_res: UnifiedPredictionResult,
        cnn_pred: Dict[str, Any],
        tree_res: Dict[str, Any],
        db_res: Dict[str, Any],
        agreement: str,
    ) -> Dict[str, Any]:
        """Build the final recommendation block from all signals."""
        # Determine primary species
        primary_species = llm_res.top_species
        primary_confidence = llm_res.top_confidence

        if primary_species in ("Unknown", "Error"):
            # Fallback to best available signal
            if tree_res.get("status") == "conclusion":
                primary_species = tree_res.get("species", "Unknown")
                primary_confidence = 0.75
            elif cnn_pred.get("conclusive"):
                primary_species = cnn_pred.get("species", "Unknown")
                primary_confidence = cnn_pred.get("confidence", 0.0)
            elif db_res.get("status") == "ok":
                primary_species = db_res.get("candidate", {}).get("english_name", "Unknown")
                primary_confidence = db_res.get("trait_match", {}).get("score", 0.0)

        # Look up species metadata
        species_meta = self._lookup_species(primary_species)

        safety_warnings = list(llm_res.safety_warnings)
        if db_res.get("safety_alert"):
            safety_warnings.append("⚠ Database safety alert: check lookalikes carefully.")

        return {
            "species": primary_species,
            "confidence": round(primary_confidence, 3),
            "swedish_name": species_meta.get("swedish_name", "Okänd"),
            "english_name": species_meta.get("english_name", primary_species),
            "scientific_name": species_meta.get("scientific_name", ""),
            "edible": species_meta.get("edible", False),
            "toxicity_level": species_meta.get("toxicity_level", "UNKNOWN"),
            "agreement": agreement,
            "needs_clarification": llm_res.needs_clarification,
            "clarification_question": llm_res.clarification_question,
            "reasoning": llm_res.reasoning,
            "safety_warnings": safety_warnings,
            "all_signals": llm_res.all_signals,
        }

    def _lookup_species(self, english_name: str) -> Dict[str, Any]:
        """Fast lookup of species metadata by English or Swedish name."""
        try:
            import csv
            with open(SPECIES_CSV, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if (row["english_name"].lower() == english_name.lower() or
                            row["swedish_name"].lower() == english_name.lower()):
                        return {
                            "swedish_name": row["swedish_name"],
                            "english_name": row["english_name"],
                            "scientific_name": row["scientific_name"],
                            "edible": row.get("edible", "FALSE").upper() == "TRUE",
                            "toxicity_level": row.get("toxicity_level", "UNKNOWN"),
                        }
        except Exception as exc:
            logger.debug("Species lookup failed: %s", exc)
        return {}
