"""
FastAPI backend for mushroom identification.

Pipeline (mirrors Sys.txt spec):
  Step 1 — Visual trait extraction: colour, shape, texture analysis → ML prediction
             + structured visible_traits dict (models/visual_trait_extractor.py)
  Step 2 — Species tree traversal via key.xml; auto-answers from Step 1 traits;
             asks user for any missing information  (models/key_tree_traversal.py)
  Step 3 — Trait database comparison + lookalike check  (models/trait_database_comparator.py)
  Step 4 — Final result: ML alternatives + lookalikes + top candidate      (models/final_aggregator.py)

Business logic helpers are in api/scoring.py.
Pydantic request models are in api/schemas.py.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    Step2AnswerRequest,
    Step2StartRequest,
    Step3CompareRequest,
    Step4FinalizeRequest,
)
from api.scoring import (
    adapt_result,
    build_prediction,
    image_scores,
    ollama_scores,
    trait_scores,
)
from models.final_aggregator import FinalAggregator
from models.hybrid_classifier import AggregationMethod, HybridClassifier
from models.key_tree_traversal import KeyTreeEngine
from models.llm_classifier import LLMClassifier, OllamaBackend
from models.trait_database_comparator import TraitDatabaseComparator

logger = logging.getLogger(__name__)


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


def _load_species_metadata() -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    with SPECIES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata[row["english_name"]] = row
    return metadata


SPECIES    = _load_species_metadata()
HYBRID     = HybridClassifier(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)
KEY_TREE   = KeyTreeEngine(str(KEY_XML))
COMPARATOR = TraitDatabaseComparator(str(DATA_RAW_DIR))
AGGREGATOR = FinalAggregator(str(SPECIES_CSV))

# Initialise Ollama LLM classifier — falls back to None if server not running
if OllamaBackend.is_available():
    try:
        LLM = LLMClassifier(backend_type="ollama")
        logger.info("Ollama LLM classifier ready")
    except Exception as _e:
        logger.warning(f"Ollama init failed, using rule-based fallback: {_e}")
        LLM = None
else:
    logger.info("Ollama not reachable — using rule-based LLM fallback (start with: ollama serve)")
    LLM = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "llm": "ollama" if LLM is not None else "rule-based",
    }


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

    img_scores, metrics, trait_extraction = image_scores(image_bytes)
    trait_based = trait_scores(trait_data)
    llm_based = ollama_scores(LLM, metrics, trait_data, trait_extraction)

    image_prediction = build_prediction(
        "image",
        trait_extraction["ml_prediction"]["reasoning"],
        img_scores,
    )
    trait_prediction = build_prediction(
        "trait",
        "Scored from questionnaire selections (cap shape, color, gill type, stem type, habitat, season).",
        trait_based,
    )
    llm_method = "ollama" if LLM is not None else "rule-based"
    llm_prediction = build_prediction(
        llm_method,
        f"LLM reasoning via {llm_method} from combined image cues and selected traits.",
        llm_based,
    )

    result = HYBRID.classify(
        image_prediction=image_prediction,
        trait_prediction=trait_prediction,
        llm_prediction=llm_prediction,
    )
    return adapt_result(result.to_dict(), metrics, trait_extraction, SPECIES)


# ---------------------------------------------------------------------------
# Step 2 — Species tree traversal (key.xml)
# ---------------------------------------------------------------------------

@app.post("/identify/Species_tree_traversal/start")
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


@app.post("/identify/Species_tree_traversal/answer")
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


@app.get("/identify/Species_tree_traversal/session/{session_id}")
def step2_session_state(session_id: str) -> Dict[str, Any]:
    """Return the current state of an active Step 2 session (for debugging)."""
    state = KEY_TREE.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return state


# ---------------------------------------------------------------------------
# Step 3 — Trait database comparison + lookalike check
# ---------------------------------------------------------------------------

@app.post("/identify/comparison/compare")
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

@app.post("/identify/prediction/finalize")
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
        body.trait_extraction_result,
        body.Species_tree_traversal_result,
        body.comparison_result,
    )

