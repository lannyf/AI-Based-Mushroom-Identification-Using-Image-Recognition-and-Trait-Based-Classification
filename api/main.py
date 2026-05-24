"""
FastAPI backend for mushroom identification.

Pipeline (mirrors Sys.txt spec):
  Step 1 — Visual trait extraction: colour, shape, texture analysis → ML prediction
             + structured visible_traits dict (models/visual_trait_extractor.py)
  Step 2 — Species tree traversal via key.xml; auto-answers from Step 1 traits;
             asks user for any missing information  (models/key_tree_traversal.py)
  Step 3 — Trait database comparison + lookalike check  (models/trait_database_comparator.py)
  Step 4 — Final result: ML alternatives + lookalikes + top candidate      (models/final_aggregator.py)

Pydantic request models are in api/schemas.py.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    LLMPredictRequest,
    Step2AnswerRequest,
    Step2StartRequest,
    Step3CompareRequest,
    Step4FinalizeRequest,
    UnifiedIdentifyRequest,
    UnifiedIdentifyResponse,
)
from models.final_aggregator import FinalAggregator
from models.key_tree_traversal import KeyTreeEngine
from models.llm_classifier import LLMClassifier, OllamaBackend
from models.trait_database_comparator import TraitDatabaseComparator
from models.cnn_classifier import get_classifier
from models.visual_trait_extractor import extract as extract_visual_traits
from models.mushroom_segmenter import Segmenter, get_segmenter
from models.unified_pipeline import UnifiedPipeline

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIES_CSV  = PROJECT_ROOT / "data" / "raw" / "species.csv"
KEY_XML      = PROJECT_ROOT / "data" / "raw" / "key.xml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# YOLO segmentation weights — prefer the fine-tuned 3-class model
YOLO_WEIGHTS = PROJECT_ROOT / "data" / "Yolov8" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "artifacts" / "yolov8_seg_ft.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "artifacts" / "yolov8n-seg.pt"

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
KEY_TREE   = KeyTreeEngine(str(KEY_XML))
COMPARATOR = TraitDatabaseComparator(str(DATA_RAW_DIR))
AGGREGATOR = FinalAggregator(str(SPECIES_CSV))

# Initialise Ollama LLM classifier — used by standalone and unified endpoints
if OllamaBackend.is_available():
    try:
        from models.key_tree_parser import KeyTreeParser
        _ktp = KeyTreeParser(str(KEY_XML))
        _key_text = _ktp.get_prompt_injection()
        LLM = LLMClassifier(backend_type="ollama", key_tree_text=_key_text)
        logger.info("Ollama LLM classifier ready (with key.xml injection)")
    except Exception as _e:
        logger.warning("Ollama init failed: %s", _e)
        LLM = None
else:
    logger.info("Ollama not reachable — LLM endpoints will return 503 (start with: ollama serve)")
    LLM = None

# Initialise unified pipeline (lazy segmenter loading)
_unified_pipeline: Optional[UnifiedPipeline] = None


def _get_unified_pipeline() -> UnifiedPipeline:
    global _unified_pipeline
    if _unified_pipeline is None:
        seg = None
        if YOLO_WEIGHTS.exists():
            try:
                seg = get_segmenter(model_path=str(YOLO_WEIGHTS))
                logger.info("UnifiedPipeline: YOLO segmenter loaded from %s", YOLO_WEIGHTS)
            except Exception as exc:
                logger.warning("UnifiedPipeline: could not load segmenter: %s", exc)
        else:
            logger.warning("UnifiedPipeline: no YOLO weights found at %s", YOLO_WEIGHTS)
        _unified_pipeline = UnifiedPipeline(
            segmenter=seg,
            key_xml_path=str(KEY_XML),
            data_raw_dir=str(DATA_RAW_DIR),
            llm_backend=LLM,
            auto_init_llm=(LLM is None),  # only auto-init if global LLM wasn't ready
        )
    return _unified_pipeline


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "llm": "ollama" if LLM is not None else "unavailable",
        "unified_pipeline": "ready" if _unified_pipeline is not None or YOLO_WEIGHTS.exists() else "no_segmenter",
    }


def _get_cnn_prediction(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Run CNN classifier on image bytes. Returns None if unavailable."""
    try:
        cnn = get_classifier()
        if not cnn.is_trained:
            return None
        cnn_scores = cnn.predict(image_bytes)
        if cnn_scores is None:
            return None
        ordered = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
        top_species, top_conf = ordered[0]
        return {
            "top_species": top_species,
            "confidence": round(top_conf, 4),
            "method": "cnn",
            "top_k": [
                {"species": sp, "confidence": round(sc, 4)}
                for sp, sc in ordered[:5]
            ],
            "reasoning": "EfficientNet-B3 CNN prediction.",
        }
    except Exception:
        return None


@app.post("/identify")
async def identify(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Step 1 — Pure visual trait extraction + optional CNN hint.

    Upload an image. The system analyses it using classical computer vision
    (colour, shape, texture, brightness) and optionally queries the CNN
    classifier for a species prediction hint.

    Returns:
      {
        "trait_extraction": {
          "visible_traits": {...},
          "ml_prediction": {...} | null
        },
        "image_analysis": {
          "red_ratio": ...,
          "orange_red_ratio": ...,
          "orange_yellow_ratio": ...,
          "brown_ratio": ...,
          "white_ratio": ...
        }
      }
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    step1 = extract_visual_traits(image_bytes)
    ml_prediction = _get_cnn_prediction(image_bytes)

    cr = step1["visible_traits"]["colour_ratios"]
    metrics = {
        "red_ratio":           cr["red"],
        "orange_red_ratio":    cr.get("orange_red", 0.0),
        "orange_yellow_ratio": cr["orange_yellow"],
        "brown_ratio":         cr["brown"],
        "white_ratio":         cr["white"],
    }

    return {
        "trait_extraction": {**step1, "ml_prediction": ml_prediction},
        "image_analysis": metrics,
    }


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

    Optionally provide ``pre_answers`` — user-supplied answers keyed by exact
    question text. These are consulted only when the image traits cannot
    provide a conclusive auto-answer.

    Response shapes:
      {"status": "question",   "session_id": ..., "question": ..., "options": [...], ...}
      {"status": "conclusion", "session_id": ..., "species": ...,  "edibility": ..., ...}
    """
    return KEY_TREE.start_session(
        body.session_id,
        body.visible_traits,
        body.ml_hint,
        body.pre_answers,
    )


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
# Step 2.5 — Standalone LLM prediction
# ---------------------------------------------------------------------------

def _build_observation_text(visible_traits: Dict[str, Any]) -> str:
    """Build a natural-language mushroom description for the LLM from extracted traits."""
    vt = visible_traits
    dominant   = vt.get("dominant_color", "unknown")
    cap_shape  = vt.get("cap_shape", "unknown")
    texture    = vt.get("surface_texture", "unknown")
    has_ridges = vt.get("has_ridges", False)

    lines = [
        f"Cap colour: {dominant}.",
        f"Cap shape: {cap_shape}, surface texture: {texture}.",
        f"Gill/underside structure: {'ridges (false gills)' if has_ridges else 'unknown'}.",
    ]
    return " ".join(lines)


@app.post("/identify/llm_predict")
def llm_predict(body: LLMPredictRequest) -> Dict[str, Any]:
    """
    Standalone LLM consultation.

    Takes only the ``visible_traits`` from Step 1, builds a natural-language
    observation, and queries the Ollama LLM for a species prediction.

    Returns 503 Service Unavailable if Ollama is not running.
    """
    if LLM is None:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Start Ollama with: ollama serve",
        )

    observation = _build_observation_text(body.visible_traits)
    result = LLM.classify(observation)

    return {
        "top_species": result.top_species,
        "confidence": round(result.top_confidence, 4),
        "reasoning": result.reasoning,
        "predictions": [
            {"species": sp, "confidence": round(conf, 4), "reason": reason}
            for sp, conf, reason in result.predictions
        ],
        "model_used": result.model_used,
        "processing_time_ms": round(result.processing_time_ms, 2),
    }


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


# ---------------------------------------------------------------------------
# Unified endpoint — two photos → full pipeline
# ---------------------------------------------------------------------------

@app.post("/identify/unified")
async def identify_unified(
    above_photo: UploadFile = File(..., description="Photo from above showing cap and stem"),
    below_photo: UploadFile = File(..., description="Photo from below showing underside and stem"),
) -> Dict[str, Any]:
    """
    Unified mushroom identification endpoint.

    Upload **two** photos of the same mushroom:
      * **above_photo** — taken from above, showing the cap and stem.
      * **below_photo** — taken from below, showing the underside (gills/pores/ridges) and stem.

    The pipeline runs automatically:
      1. YOLOv8 3-class segmentation on both images
      2. Case detection (classical / puffball / uncertain)
      3. Trait extraction per photo (masked where possible)
      4. CNN prediction with uncertainty flags
      5. Key-tree auto-traversal
      6. Trait-database comparison
      7. LLM synthesis of all signals
      8. Agreement evaluation

    Returns a complete result dict with all intermediate outputs and a
    ``final_recommendation`` block.
    """
    above_bytes = await above_photo.read()
    below_bytes = await below_photo.read()

    if not above_bytes or not below_bytes:
        raise HTTPException(
            status_code=400,
            detail="Both above_photo and below_photo must be provided.",
        )

    pipeline = _get_unified_pipeline()
    result = pipeline.run(above_bytes, below_bytes)
    return result
