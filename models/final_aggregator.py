"""
Step 4 — Final Result Aggregator

Receives the outputs of Steps 1, 2, and 3 and produces a single structured
final answer for the user, as described in Sys.txt:

  "The LLM should then present:
     - The alternatives given from the machine learning (Step 1)
     - The exchangeable species from the mushroom dataset (Step 3 lookalikes)
     - Its own most likely top candidate"

Confidence weighting
--------------------
  Step 2 (key.xml tree traversal) is the primary authority — it follows a
  validated expert key.  Weight: 0.45

  Step 1 (image analysis) provides a corroborating signal.  Weight: 0.35

  Step 3 (trait database match) validates morphology.  Weight: 0.20

  When Step 2 and Step 1 agree on the same species the overall confidence
  receives a +10 % agreement bonus (capped at 1.0).

Public API
----------
  aggregator = FinalAggregator(species_csv_path)
  result = aggregator.aggregate(step1, step2, step3)

  # Result shape:
  # {
  #   "final_recommendation": {
  #       species_id, swedish_name, english_name, scientific_name,
  #       edible, toxicity_level,
  #       overall_confidence,
  #       confidence_breakdown: {image, tree_traversal, trait_match},
  #       reasoning
  #   },
  #   "ml_alternatives":      [{species, confidence, swedish_name, english_name}],
  #   "exchangeable_species":  [{full lookalike info from Step 3}],
  #   "safety_warnings":       [str],
  #   "verdict":               "edible" | "inedible" | "toxic" | "unknown",
  #   "method_agreement":      "full" | "partial" | "none"
  # }
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Species name → species_id lookup for Step 1 english names
# ---------------------------------------------------------------------------

_STEP1_NAME_TO_ID: Dict[str, str] = {
    "Fly Agaric":        "AM.MU",
    "Chanterelle":       "CA.CI",
    "False Chanterelle": "HY.PS",
    "Porcini":           "BO.ED",
    "Black Trumpet":     "CR.CO",
    "Other Boletus":     "BO.ED",   # fallback
    "Amanita virosa":    "AM.VI",
}


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------

def _make_verdict(toxicity: str, edible: bool) -> str:
    tox = toxicity.upper()
    if tox in {"EXTREMELY_TOXIC", "TOXIC"}:
        return "toxic"
    if tox == "PSYCHOACTIVE":
        return "toxic"
    if edible:
        return "edible"
    return "inedible"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class FinalAggregator:
    """
    Combines Step 1 / Step 2 / Step 3 outputs into a single final answer.
    """

    def __init__(self, species_csv_path: str) -> None:
        self._species_by_id:   Dict[str, Dict[str, str]] = {}
        self._species_by_name: Dict[str, Dict[str, str]] = {}
        with Path(species_csv_path).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self._species_by_id[row["species_id"]]        = row
                self._species_by_name[row["english_name"].lower()] = row
                self._species_by_name[row["swedish_name"].lower()] = row
        logger.info("FinalAggregator loaded %d species", len(self._species_by_id))

    # ------------------------------------------------------------------
    def aggregate(
        self,
        step1: Dict[str, Any],
        step2: Dict[str, Any],
        step3: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce the Step 4 final answer.

        Args:
            step1: Full output of POST /identify  (must contain 'step1' key
                   with 'ml_prediction' and 'visible_traits')
            step2: Output of POST /identify/step2/start or /step2/answer
                   when status == 'conclusion'
            step3: Output of POST /identify/step3/compare
        """
        # ---- Extract sub-objects ----------------------------------------
        s1_ml   = step1.get("step1", step1).get("ml_prediction", {})
        s2_ok   = step2.get("status") == "conclusion"
        s3_ok   = step3.get("status") == "ok"

        # ---- Determine primary candidate --------------------------------
        # Priority: Step 2 (expert key) > Step 3 DB resolution > Step 1 image
        candidate_row: Optional[Dict[str, str]] = None
        primary_source = "unknown"

        if s3_ok and step3.get("candidate"):
            sid = step3["candidate"]["species_id"]
            candidate_row = self._species_by_id.get(sid)
            primary_source = "tree_traversal+db"

        if candidate_row is None and s2_ok:
            # Try to look up by species name from Step 2
            cname = step2.get("species", "").lower()
            candidate_row = self._species_by_name.get(cname)
            primary_source = "tree_traversal"

        if candidate_row is None:
            # Fall back to Step 1 top ML species
            ml_top = s1_ml.get("top_species", "")
            sid = _STEP1_NAME_TO_ID.get(ml_top)
            if sid:
                candidate_row = self._species_by_id.get(sid)
            primary_source = "image_analysis"

        # ---- Confidence calculation ------------------------------------
        # Step 2 confidence: 1.0 if conclusion reached, else 0.0
        tree_conf = 1.0 if s2_ok else 0.0

        # Step 1 confidence for the top species
        image_conf = float(s1_ml.get("confidence", 0.0))

        # Step 3 trait match score
        trait_conf = float(
            step3.get("trait_match", {}).get("score", 0.0)
        ) if s3_ok else 0.5  # neutral if Step 3 not available

        # Weighted combination
        W_TREE  = 0.45
        W_IMAGE = 0.35
        W_TRAIT = 0.20
        overall = W_TREE * tree_conf + W_IMAGE * image_conf + W_TRAIT * trait_conf

        # Agreement bonus: +10 % when Steps 1 and 2 agree on the same species
        agreement = "none"
        if s2_ok and candidate_row:
            s2_species = step2.get("species", "").lower()
            s1_top_id  = _STEP1_NAME_TO_ID.get(s1_ml.get("top_species", ""), "")
            if (candidate_row["species_id"] == s1_top_id or
                    candidate_row["swedish_name"].lower() in s2_species or
                    s2_species in candidate_row["swedish_name"].lower()):
                overall = min(1.0, overall + 0.10)
                agreement = "full"
            elif tree_conf > 0 or image_conf > 0.2:
                agreement = "partial"
        elif tree_conf > 0 or image_conf > 0.2:
            agreement = "partial"

        overall = round(overall, 3)

        # ---- Build reasoning text ------------------------------------
        reasoning_parts = []
        if s1_ml.get("reasoning"):
            reasoning_parts.append(f"Image analysis: {s1_ml['reasoning']}")
        if s2_ok:
            path = " → ".join(step2.get("path", []))
            auto = step2.get("auto_answered", [])
            manual = len(step2.get("path", [])) - len(auto)
            reasoning_parts.append(
                f"Species key traversal concluded '{step2.get('species', '')}' "
                f"({len(auto)} steps auto-resolved from image, {manual} answered by user)."
            )
        if s3_ok and step3.get("trait_match"):
            tm = step3["trait_match"]
            n_match = len(tm.get("matched", []))
            n_conf  = len(tm.get("conflicts", []))
            reasoning_parts.append(
                f"Trait database match: {tm['score']:.0%} "
                f"({n_match} traits matched, {n_conf} conflicts)."
            )

        # ---- ML alternatives (Step 1 top-k) ---------------------------
        ml_alternatives = []
        for entry in s1_ml.get("top_k", []):
            sp_name = entry.get("species", "")
            sid = _STEP1_NAME_TO_ID.get(sp_name)
            db_row = self._species_by_id.get(sid) if sid else None
            ml_alternatives.append({
                "species":      sp_name,
                "confidence":   round(float(entry.get("confidence", 0)), 4),
                "swedish_name": db_row["swedish_name"] if db_row else sp_name,
                "english_name": db_row["english_name"] if db_row else sp_name,
                "species_id":   sid or "—",
            })

        # ---- Exchangeable species (Step 3 lookalikes) -----------------
        exchangeable = step3.get("lookalikes", []) if s3_ok else []

        # ---- Safety warnings ------------------------------------------
        safety_warnings: List[str] = []
        if candidate_row:
            tox = candidate_row.get("toxicity_level", "SAFE").upper()
            if tox in {"TOXIC", "EXTREMELY_TOXIC"}:
                safety_warnings.append(
                    f"⚠ DANGER: {candidate_row['swedish_name']} is POISONOUS. "
                    "Do not consume."
                )

        for la in exchangeable:
            if la.get("safety_alert"):
                safety_warnings.append(
                    f"⚠ Easily confused with {la['swedish_name']} "
                    f"({la.get('toxicity_level','?')}) — "
                    f"{la.get('confusion_likelihood','?')} confusion risk. "
                    + (la.get("distinguishing_features", "")[:120] or "")
                )

        if step3.get("safety_alert") and not safety_warnings:
            safety_warnings.append(
                "⚠ Safety alert: check lookalikes carefully before consuming."
            )

        # ---- Final recommendation block --------------------------------
        if candidate_row:
            edible = candidate_row.get("edible", "FALSE").upper() == "TRUE"
            toxicity = candidate_row.get("toxicity_level", "UNKNOWN")
            final_rec: Dict[str, Any] = {
                "species_id":      candidate_row["species_id"],
                "swedish_name":    candidate_row["swedish_name"],
                "english_name":    candidate_row["english_name"],
                "scientific_name": candidate_row["scientific_name"],
                "edible":          edible,
                "toxicity_level":  toxicity,
                "overall_confidence": overall,
                "confidence_breakdown": {
                    "image_analysis":  round(image_conf, 3),
                    "tree_traversal":  round(tree_conf, 3),
                    "trait_match":     round(trait_conf, 3),
                },
                "primary_source": primary_source,
                "reasoning":       " ".join(reasoning_parts) or "Identification based on available signals.",
            }
            verdict = _make_verdict(toxicity, edible)
        else:
            final_rec = {
                "species_id":   "unknown",
                "swedish_name": "Okänd",
                "english_name": "Unknown",
                "overall_confidence": overall,
                "confidence_breakdown": {
                    "image_analysis": round(image_conf, 3),
                    "tree_traversal": round(tree_conf, 3),
                    "trait_match":    round(trait_conf, 3),
                },
                "primary_source": primary_source,
                "reasoning": "Could not resolve species from available data.",
            }
            verdict = "unknown"
            safety_warnings.append(
                "⚠ Could not identify species — do not consume."
            )

        return {
            "final_recommendation": final_rec,
            "ml_alternatives":      ml_alternatives,
            "exchangeable_species": exchangeable,
            "safety_warnings":      safety_warnings,
            "verdict":              verdict,
            "method_agreement":     agreement,
        }
