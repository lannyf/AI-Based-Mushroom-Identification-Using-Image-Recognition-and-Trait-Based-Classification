"""
FastAPI backend for mushroom identification.

This uses the existing HybridClassifier aggregation logic and combines
deterministic image/trait heuristics into real API responses that the Flutter
app can consume. It is a practical inference backend for local development
until trained model artifacts are wired in.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from models.hybrid_classifier import AggregationMethod, HybridClassifier, MethodPrediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIES_CSV = PROJECT_ROOT / "data" / "raw" / "species.csv"

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


SPECIES = load_species_metadata()
HYBRID = HybridClassifier(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)

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


def image_scores(image_bytes: bytes) -> Tuple[Dict[str, float], Dict[str, float]]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    red_ratio = float(np.mean((r > 150) & (r > g * 1.25) & (r > b * 1.25)))
    orange_yellow_ratio = float(np.mean((r > 140) & (g > 90) & (b < 140)))
    brown_ratio = float(np.mean((r > 80) & (g > 45) & (b < 90) & (r > g) & (g > b)))
    white_ratio = float(np.mean((r > 185) & (g > 185) & (b > 185)))

    scores = {name: 0.02 for name in TARGET_SPECIES}
    scores["Fly Agaric"] += red_ratio * 5.0 + white_ratio * 1.4
    scores["Amanita virosa"] += white_ratio * 2.5
    scores["Chanterelle"] += orange_yellow_ratio * 4.0
    scores["False Chanterelle"] += orange_yellow_ratio * 2.6
    scores["Porcini"] += brown_ratio * 4.2
    scores["Other Boletus"] += brown_ratio * 3.3
    scores["Black Trumpet"] += (1.0 - white_ratio) * 0.2 + orange_yellow_ratio * 0.4

    metrics = {
        "red_ratio": red_ratio,
        "orange_yellow_ratio": orange_yellow_ratio,
        "brown_ratio": brown_ratio,
        "white_ratio": white_ratio,
    }
    return scores, metrics


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


def adapt_result(result: Dict[str, Any], image_metrics: Dict[str, float]) -> Dict[str, Any]:
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

    img_scores, metrics = image_scores(image_bytes)
    trait_based = trait_scores(trait_data)
    llm_based = llm_scores(metrics, trait_data)

    image_prediction = build_prediction(
        "image",
        f"Color profile red={metrics['red_ratio']:.2f}, yellow/orange={metrics['orange_yellow_ratio']:.2f}, "
        f"brown={metrics['brown_ratio']:.2f}, white={metrics['white_ratio']:.2f}",
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
    return adapt_result(result.to_dict(), metrics)
