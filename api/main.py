"""
FastAPI backend for mushroom identification.

Pipeline (mirrors Sys.txt spec):
  Step 1 — Visual trait extraction: colour, shape, texture analysis → ML prediction
             + structured visible_traits dict (models/visual_trait_extractor.py)
  Step 2 — Species tree traversal via key.xml; auto-answers from Step 1 traits;
             asks user for any missing information  (models/key_tree_traversal.py)
  Step 3 — Trait database comparison + lookalike check  (models/trait_database_comparator.py)
  Step 4 — Final result: ML alternatives + lookalikes + top candidate      (models/final_aggregator.py)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.final_aggregator import FinalAggregator
from models.hybrid_classifier import AggregationMethod, HybridClassifier, MethodPrediction
from models.key_tree_traversal import KeyTreeEngine
from models.trait_database_comparator import TraitDatabaseComparator
from models.visual_trait_extractor import extract as extract_visual_traits


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIES_CSV  = PROJECT_ROOT / "data" / "raw" / "species.csv"
KEY_XML      = PROJECT_ROOT / "data" / "raw" / "key.xml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

app = FastAPI(title="Mushroom Identification API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_species_metadata() -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    with SPECIES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata[row["english_name"]] = row
    return metadata


SPECIES    = load_species_metadata()
HYBRID     = HybridClassifier(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)
KEY_TREE   = KeyTreeEngine(str(KEY_XML))
COMPARATOR = TraitDatabaseComparator(str(DATA_RAW_DIR))
AGGREGATOR = FinalAggregator(str(SPECIES_CSV))

TARGET_SPECIES = [
    "Fly Agaric",
    "Chanterelle",
    "False Chanterelle",
    "Porcini",
    "Other Boletus",
    "Amanita virosa",
    "Black Trumpet",
]


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(value, 0.0) for value in scores.values())
    if total <= 0:
        even = 1.0 / len(scores)
        return {key: even for key in scores}
    return {key: max(value, 0.0) / total for key, value in scores.items()}


def build_prediction(method: str, reasoning: str, scores: Dict[str, float]) -> MethodPrediction:
    normalized = normalize(scores)
    ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    species, confidence = ordered[0]
    return MethodPrediction(
        method=method,
        species=species,
        confidence=float(confidence),
        reasoning=reasoning,
        top_k=[(name, float(score)) for name, score in ordered[:5]],
    )


def image_scores(image_bytes: bytes) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Step 1 — Visual analysis.

    Returns:
      scores        — per-species unnormalised scores (for HybridClassifier)
      metrics       — legacy colour-ratio dict (kept for llm_scores compatibility)
      step1_result  — full Step-1 output: ml_prediction + visible_traits
    """
    step1 = extract_visual_traits(image_bytes)
    vt = step1["visible_traits"]
    cr = vt["colour_ratios"]

    # Build scores dict from the extractor's own top-k
    scores: Dict[str, float] = {name: 0.02 for name in TARGET_SPECIES}
    for entry in step1["ml_prediction"]["top_k"]:
        if entry["species"] in scores:
            scores[entry["species"]] = entry["confidence"]

    # Legacy metrics dict (used by llm_scores below)
    metrics = {
        "red_ratio":            cr["red"],
        "orange_yellow_ratio":  cr["orange_yellow"],
        "brown_ratio":          cr["brown"],
        "white_ratio":          cr["white"],
    }
    return scores, metrics, step1


def trait_scores(traits: Dict[str, Any]) -> Dict[str, float]:
    color = str(traits.get("color", "")).lower()
    cap_shape = str(traits.get("cap_shape", "")).lower()
    gill_type = str(traits.get("gill_type", "")).lower()
    stem_type = str(traits.get("stem_type", "")).lower()
    habitat = str(traits.get("habitat", "")).lower()
    season = str(traits.get("season", "")).lower()

    scores = {name: 0.05 for name in TARGET_SPECIES}

    if color == "red":
        scores["Fly Agaric"] += 0.8
        scores["Amanita virosa"] += 0.1
    if color in {"orange", "yellow"}:
        scores["Chanterelle"] += 0.7
        scores["False Chanterelle"] += 0.45
        scores["Black Trumpet"] += 0.15
    if color == "brown":
        scores["Porcini"] += 0.7
        scores["Other Boletus"] += 0.5
    if color == "white":
        scores["Amanita virosa"] += 0.55

    if cap_shape == "wavy":
        scores["Chanterelle"] += 0.45
        scores["False Chanterelle"] += 0.2
    if cap_shape in {"convex", "flat"}:
        scores["Porcini"] += 0.25
        scores["Other Boletus"] += 0.2
    if cap_shape in {"bell-shaped", "umbrella-shaped"}:
        scores["Fly Agaric"] += 0.35
        scores["Amanita virosa"] += 0.25

    if gill_type == "decurrent":
        scores["Chanterelle"] += 0.7
        scores["False Chanterelle"] += 0.25
    if gill_type == "free":
        scores["Fly Agaric"] += 0.45
        scores["Amanita virosa"] += 0.45
    if gill_type == "attached":
        scores["Porcini"] += 0.15
        scores["Other Boletus"] += 0.15

    if stem_type == "bulbous":
        scores["Porcini"] += 0.3
        scores["Fly Agaric"] += 0.35
        scores["Amanita virosa"] += 0.3
    if stem_type in {"ring/annulus", "cup/volva"}:
        scores["Fly Agaric"] += 0.65
        scores["Amanita virosa"] += 0.65
    if stem_type == "solid":
        scores["Chanterelle"] += 0.15
        scores["Porcini"] += 0.15

    if habitat == "forest":
        scores["Fly Agaric"] += 0.18
        scores["Chanterelle"] += 0.18
        scores["Porcini"] += 0.18
    if habitat == "dead wood":
        scores["Black Trumpet"] += 0.08

    if season in {"summer", "autumn", "fall"}:
        scores["Chanterelle"] += 0.08
        scores["Porcini"] += 0.08
        scores["Fly Agaric"] += 0.08

    return scores


def llm_scores(image_metrics: Dict[str, float], traits: Dict[str, Any]) -> Dict[str, float]:
    scores = {name: 0.03 for name in TARGET_SPECIES}
    color = str(traits.get("color", "")).lower()
    gill_type = str(traits.get("gill_type", "")).lower()
    stem_type = str(traits.get("stem_type", "")).lower()

    if image_metrics["red_ratio"] > 0.08 and image_metrics["white_ratio"] > 0.03:
        scores["Fly Agaric"] += 0.9
        scores["Amanita virosa"] += 0.15
    elif color in {"orange", "yellow"} and gill_type == "decurrent":
        scores["Chanterelle"] += 0.82
        scores["False Chanterelle"] += 0.22
    elif color == "brown":
        scores["Porcini"] += 0.78
        scores["Other Boletus"] += 0.35
    elif stem_type in {"ring/annulus", "cup/volva"}:
        scores["Fly Agaric"] += 0.5
        scores["Amanita virosa"] += 0.45
    else:
        scores["Porcini"] += 0.25
        scores["Chanterelle"] += 0.25

    return scores


def safety_rating(species: str) -> str:
    metadata = SPECIES.get(species)
    if not metadata:
        return "unknown"

    toxicity = metadata.get("toxicity_level", "UNKNOWN").upper()
    edible = metadata.get("edible", "FALSE").upper() == "TRUE"

    if toxicity in {"EXTREMELY_TOXIC", "TOXIC"} or not edible:
        return "inedible"
    if edible:
        return "edible"
    return "unknown"


def common_name_for(species: str) -> Tuple[str, str]:
    """Return (english_name, swedish_name) for a given species key."""
    metadata = SPECIES.get(species)
    if not metadata:
        return species, species
    return metadata.get("english_name", species), metadata.get("swedish_name", species)


def adapt_result(
    result: Dict[str, Any],
    image_metrics: Dict[str, float],
    step1: Dict[str, Any],
) -> Dict[str, Any]:
    predictions = [
        {
            "species": item["species"],
            "confidence": item["confidence"],
            "common": common_name_for(item["species"])[0],
            "swedish_name": common_name_for(item["species"])[1],
        }
        for item in result["predictions"]
    ]

    lookalikes = []
    for item in result["lookalikes"]:
        similarity = float(item["similarity"])
        risk = "high" if similarity >= 0.8 else "medium" if similarity >= 0.6 else "low"
        lookalikes.append(
            {
                "species": item["species"],
                "risk": risk,
                "reason": item["reason"],
            }
        )

    top_species = result["top_species"]

    return {
        # --- Step 1: what the image analysis found ---
        "step1": step1,

        # --- Final aggregated result ---
        "top_prediction": top_species,
        "overall_confidence": result["confidence"],
        "method_confidences": result["confidence_breakdown"],
        "predictions": predictions,
        "top_predictions": predictions,
        "lookalikes": lookalikes,
        "safety_rating": safety_rating(top_species),
        "confidence": result["confidence"],
        "safety_warnings": result["safety_warnings"],
        "aggregation_method": result["aggregation_method"],
        "consensus_strength": result["consensus_strength"],
        "image_analysis": image_metrics,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/identify")
async def identify(
    image: UploadFile = File(...),
    traits: str = Form("{}"),
) -> Dict[str, Any]:
    try:
        trait_data = json.loads(traits) if traits else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid traits JSON: {exc}") from exc

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    img_scores, metrics, step1 = image_scores(image_bytes)
    trait_based = trait_scores(trait_data)
    llm_based = llm_scores(metrics, trait_data)

    image_prediction = build_prediction(
        "image",
        step1["ml_prediction"]["reasoning"],
        img_scores,
    )
    trait_prediction = build_prediction(
        "trait",
        "Scored from questionnaire selections (cap shape, color, gill type, stem type, habitat, season).",
        trait_based,
    )
    llm_prediction = build_prediction(
        "llm",
        "Rule-based expert reasoning derived from combined image cues and selected traits.",
        llm_based,
    )

    result = HYBRID.classify(
        image_prediction=image_prediction,
        trait_prediction=trait_prediction,
        llm_prediction=llm_prediction,
    )
    return adapt_result(result.to_dict(), metrics, step1)


# ---------------------------------------------------------------------------
# Step 2 — Species tree traversal (key.xml)
# ---------------------------------------------------------------------------

class Step2StartRequest(BaseModel):
    session_id: Optional[str] = None
    visible_traits: Dict[str, Any]      # output of Step 1 visible_traits


class Step2AnswerRequest(BaseModel):
    session_id: str
    answer: str                          # one of the options returned by the previous call


@app.post("/identify/step2/start")
def step2_start(body: Step2StartRequest) -> Dict[str, Any]:
    """
    Begin Step 2 traversal.

    Post the ``visible_traits`` dict from the Step 1 ``/identify`` response.
    The engine will auto-answer as many questions as it can from the image data
    and return the first question that requires human input, or a conclusion
    if the tree can be fully resolved automatically.

    Response shapes:
      {"status": "question",   "session_id": ..., "question": ..., "options": [...], ...}
      {"status": "conclusion", "session_id": ..., "species": ...,  "edibility": ..., ...}
    """
    return KEY_TREE.start_session(body.session_id, body.visible_traits)


@app.post("/identify/step2/answer")
def step2_answer(body: Step2AnswerRequest) -> Dict[str, Any]:
    """
    Provide the user's answer to the current question and continue traversal.

    ``answer`` must exactly match one of the ``options`` returned by the
    previous ``/step2/start`` or ``/step2/answer`` call.

    Continues until a conclusion (species) is reached.
    """
    result = KEY_TREE.answer(body.session_id, body.answer)
    if result.get("status") == "error" and "Session not found" in result.get("message", ""):
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@app.get("/identify/step2/session/{session_id}")
def step2_session_state(session_id: str) -> Dict[str, Any]:
    """Return the current state of an active Step 2 session (for debugging)."""
    state = KEY_TREE.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return state


# ---------------------------------------------------------------------------
# Step 3 — Trait database comparison + lookalike check
# ---------------------------------------------------------------------------

class Step3CompareRequest(BaseModel):
    swedish_name: str             # species name from Step 2 conclusion
    visible_traits: Dict[str, Any]  # Step 1 visible_traits


@app.post("/identify/step3/compare")
def step3_compare(body: Step3CompareRequest) -> Dict[str, Any]:
    """
    Step 3 — Trait database comparison.

    Post the species name from the Step 2 conclusion and the Step 1
    visible_traits. The engine will:
      1. Resolve the Swedish name to a database species record.
      2. Compare visible traits against the species' stored trait profile.
      3. List all confusable lookalike species with distinguishing features.
      4. Flag a safety_alert if any lookalike is toxic/deadly.

    Response:
      {
        "status":           "ok" | "species_not_found",
        "candidate":        {species_id, swedish_name, english_name, ...},
        "name_match_score": float,
        "trait_match":      {score, matched, conflicts, not_comparable},
        "lookalikes":       [{species info + trait_differences + safety_alert}],
        "safety_alert":     bool
      }
    """
    return COMPARATOR.compare(body.swedish_name, body.visible_traits)


# ---------------------------------------------------------------------------
# Step 4 — Final aggregation and presentation
# ---------------------------------------------------------------------------

class Step4FinalizeRequest(BaseModel):
    step1_result: Dict[str, Any]   # full /identify response (contains "step1" key)
    step2_result: Dict[str, Any]   # /step2/start or /step2/answer when status=conclusion
    step3_result: Dict[str, Any]   # /step3/compare response


@app.post("/identify/step4/finalize")
def step4_finalize(body: Step4FinalizeRequest) -> Dict[str, Any]:
    """
    Step 4 — Final result aggregation.

    Combines the outputs of Steps 1, 2, and 3 into a single structured
    answer for the user, as specified in Sys.txt:

      - ML alternatives from image analysis (Step 1)
      - Exchangeable / confusable species from the database (Step 3)
      - The system's own top candidate with overall confidence

    Confidence is a weighted combination:
      Step 2 (expert key traversal): 45 %
      Step 1 (image analysis):       35 %
      Step 3 (trait match):          20 %
    A +10 % agreement bonus is applied when Steps 1 and 2 agree.

    Response:
      {
        "final_recommendation": {
            species_id, swedish_name, english_name, scientific_name,
            edible, toxicity_level, overall_confidence,
            confidence_breakdown, reasoning
        },
        "ml_alternatives":      [{species, confidence, swedish_name, ...}],
        "exchangeable_species": [{full lookalike info from Step 3}],
        "safety_warnings":      [str],
        "verdict":              "edible" | "inedible" | "toxic" | "unknown",
        "method_agreement":     "full" | "partial" | "none"
      }
    """
    return AGGREGATOR.aggregate(
        body.step1_result,
        body.step2_result,
        body.step3_result,
    )

