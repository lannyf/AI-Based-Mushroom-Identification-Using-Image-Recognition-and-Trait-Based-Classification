"""
Pure scoring and result-building helpers for the mushroom identification API.

Extracted from api/main.py to keep the route module focused on HTTP concerns.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models.hybrid_classifier import MethodPrediction
from models.visual_trait_extractor import extract as extract_visual_traits


TARGET_SPECIES: List[str] = [
    "Fly Agaric",
    "Chanterelle",
    "False Chanterelle",
    "Porcini",
    "Other Boletus",
    "Amanita virosa",
    "Black Trumpet",
]


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalise a score dict so values sum to 1, clipping negatives to 0."""
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        even = 1.0 / len(scores)
        return {k: even for k in scores}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


# ---------------------------------------------------------------------------
# MethodPrediction builder
# ---------------------------------------------------------------------------

def build_prediction(
    method: str,
    reasoning: str,
    scores: Dict[str, float],
) -> MethodPrediction:
    """Normalise *scores*, pick the top species, and wrap in a MethodPrediction."""
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


# ---------------------------------------------------------------------------
# Step 1 — image analysis
# ---------------------------------------------------------------------------

def image_scores(
    image_bytes: bytes,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Run Step-1 visual analysis.

    Returns:
      scores       — per-species scores for HybridClassifier
      metrics      — legacy colour-ratio dict used by llm_scores
      trait_extraction_result — full Step-1 output (ml_prediction + visible_traits)
    """
    step1 = extract_visual_traits(image_bytes)
    vt = step1["visible_traits"]
    cr = vt["colour_ratios"]

    scores: Dict[str, float] = {name: 0.02 for name in TARGET_SPECIES}
    for entry in step1["ml_prediction"]["top_k"]:
        if entry["species"] in scores:
            scores[entry["species"]] = entry["confidence"]

    metrics = {
        "red_ratio":           cr["red"],
        "orange_red_ratio":    cr.get("orange_red", 0.0),
        "orange_yellow_ratio": cr["orange_yellow"],
        "brown_ratio":         cr["brown"],
        "white_ratio":         cr["white"],
    }
    return scores, metrics, step1


# ---------------------------------------------------------------------------
# Step 1 — trait-based scoring
# ---------------------------------------------------------------------------

def trait_scores(traits: Dict[str, Any]) -> Dict[str, float]:
    """Score each target species from questionnaire trait selections."""
    color = str(traits.get("color", "")).lower()
    cap_shape = str(traits.get("cap_shape", "")).lower()
    gill_type = str(traits.get("gill_type", "")).lower()
    stem_type = str(traits.get("stem_type", "")).lower()
    habitat = str(traits.get("habitat", "")).lower()
    season = str(traits.get("season", "")).lower()

    scores: Dict[str, float] = {name: 0.05 for name in TARGET_SPECIES}

    # colour signals
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

    # cap shape signals
    if cap_shape == "wavy":
        scores["Chanterelle"] += 0.45
        scores["False Chanterelle"] += 0.2
    if cap_shape in {"convex", "flat"}:
        scores["Porcini"] += 0.25
        scores["Other Boletus"] += 0.2
    if cap_shape in {"bell-shaped", "umbrella-shaped"}:
        scores["Fly Agaric"] += 0.35
        scores["Amanita virosa"] += 0.25

    # gill type signals
    if gill_type == "decurrent":
        scores["Chanterelle"] += 0.7
        scores["False Chanterelle"] += 0.25
    if gill_type == "free":
        scores["Fly Agaric"] += 0.45
        scores["Amanita virosa"] += 0.45
    if gill_type == "attached":
        scores["Porcini"] += 0.15
        scores["Other Boletus"] += 0.15

    # stem type signals
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

    # habitat signals
    if habitat == "forest":
        scores["Fly Agaric"] += 0.18
        scores["Chanterelle"] += 0.18
        scores["Porcini"] += 0.18
    if habitat == "dead wood":
        scores["Black Trumpet"] += 0.08

    # season signals
    if season in {"summer", "autumn", "fall"}:
        scores["Chanterelle"] += 0.08
        scores["Porcini"] += 0.08
        scores["Fly Agaric"] += 0.08

    return scores


# ---------------------------------------------------------------------------
# Step 1 — rule-based LLM-style scoring
# ---------------------------------------------------------------------------

def llm_scores(
    image_metrics: Dict[str, float],
    traits: Dict[str, Any],
) -> Dict[str, float]:
    """Rule-based expert scoring derived from image colour cues and traits."""
    scores: Dict[str, float] = {name: 0.03 for name in TARGET_SPECIES}
    color = str(traits.get("color", "")).lower()
    gill_type = str(traits.get("gill_type", "")).lower()
    stem_type = str(traits.get("stem_type", "")).lower()

    if (image_metrics["red_ratio"] > 0.08
            or image_metrics.get("orange_red_ratio", 0.0) > 0.08) \
            and image_metrics["white_ratio"] > 0.02:
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


def build_observation_text(
    metrics: Dict[str, float],
    traits: Dict[str, Any],
    trait_extraction: Dict[str, Any],
) -> str:
    """Build a natural-language mushroom description for the LLM from extracted data."""
    vt = trait_extraction.get("visible_traits", {})
    dominant   = vt.get("dominant_color", traits.get("color", "unknown"))
    cap_shape  = vt.get("cap_shape",  traits.get("cap_shape",  "unknown"))
    texture    = vt.get("surface_texture", "unknown")
    has_ridges = vt.get("has_ridges", False)
    gill_type  = traits.get("gill_type", "unknown")
    stem_type  = traits.get("stem_type", "unknown")
    habitat    = traits.get("habitat",   "forest")
    season     = traits.get("season",    "autumn")

    lines = [
        f"Cap colour: {dominant}.",
        f"Cap shape: {cap_shape}, surface texture: {texture}.",
        f"Gill/underside structure: {'ridges (false gills)' if has_ridges else gill_type}.",
        f"Stem type: {stem_type}.",
        f"Habitat: {habitat}, season: {season}.",
        f"Colour analysis — red: {metrics.get('red_ratio', 0):.2f}, "
        f"orange-red: {metrics.get('orange_red_ratio', 0):.2f}, "
        f"orange-yellow: {metrics.get('orange_yellow_ratio', 0):.2f}, "
        f"brown: {metrics.get('brown_ratio', 0):.2f}, "
        f"white: {metrics.get('white_ratio', 0):.2f}.",
    ]
    return " ".join(lines)


def ollama_scores(
    classifier: Any,
    metrics: Dict[str, float],
    traits: Dict[str, Any],
    trait_extraction: Dict[str, Any],
) -> Dict[str, float]:
    """
    Query the Ollama LLM and convert the result to a per-species score dict.
    Falls back to rule-based llm_scores() if Ollama is unavailable or errors.
    """
    if classifier is None:
        return llm_scores(metrics, traits)

    try:
        observation = build_observation_text(metrics, traits, trait_extraction)
        result = classifier.classify(observation)
        # Convert (species, confidence, reason) tuples to scores dict
        scores: Dict[str, float] = {name: 0.01 for name in TARGET_SPECIES}
        for species, confidence, _ in result.predictions:
            if species in scores:
                scores[species] = float(confidence)
        # If top prediction is a target species, boost it directly
        if result.top_species in scores:
            scores[result.top_species] = max(scores[result.top_species], result.top_confidence)
        return scores
    except Exception:
        # Graceful fallback — Ollama may be slow, offline, or model not pulled yet
        return llm_scores(metrics, traits)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def safety_rating(species: str, species_metadata: Dict[str, Dict[str, str]]) -> str:
    """Return 'edible', 'inedible', or 'unknown' for a species name."""
    metadata = species_metadata.get(species)
    if not metadata:
        return "unknown"
    toxicity = metadata.get("toxicity_level", "UNKNOWN").upper()
    edible = metadata.get("edible", "FALSE").upper() == "TRUE"
    if toxicity in {"EXTREMELY_TOXIC", "TOXIC"} or not edible:
        return "inedible"
    if edible:
        return "edible"
    return "unknown"


def common_name_for(
    species: str,
    species_metadata: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    """Return (english_name, swedish_name) for a given species key."""
    metadata = species_metadata.get(species)
    if not metadata:
        return species, species
    return metadata.get("english_name", species), metadata.get("swedish_name", species)


def adapt_result(
    result: Dict[str, Any],
    image_metrics: Dict[str, float],
    trait_extraction: Dict[str, Any],
    species_metadata: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Convert a HybridClassifier result dict into the API response shape."""
    predictions = [
        {
            "species": item["species"],
            "confidence": item["confidence"],
            "common": common_name_for(item["species"], species_metadata)[0],
            "swedish_name": common_name_for(item["species"], species_metadata)[1],
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
        "trait_extraction": trait_extraction,
        "top_prediction": top_species,
        "overall_confidence": result["confidence"],
        "method_confidences": result["confidence_breakdown"],
        "predictions": predictions,
        "top_predictions": predictions,
        "lookalikes": lookalikes,
        "safety_rating": safety_rating(top_species, species_metadata),
        "confidence": result["confidence"],
        "safety_warnings": result["safety_warnings"],
        "aggregation_method": result["aggregation_method"],
        "consensus_strength": result["consensus_strength"],
        "image_analysis": image_metrics,
    }
