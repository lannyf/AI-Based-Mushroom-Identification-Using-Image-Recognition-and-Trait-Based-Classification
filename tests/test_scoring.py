"""
Unit tests for api/scoring.py

Tests cover:
  - normalize: sums to 1.0, handles edge cases
  - build_prediction: returns correct MethodPrediction with top species
  - trait_scores: colour / cap / gill / stem / habitat / season rules
  - llm_scores: image-metric + trait rule combinations
  - safety_rating: edible / inedible / unknown resolution
  - common_name_for: name lookup
  - adapt_result: output shape and field values
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from api.scoring import (
    adapt_result,
    build_prediction,
    common_name_for,
    llm_scores,
    normalize,
    safety_rating,
    trait_scores,
)


# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

FAKE_METADATA: Dict[str, Dict[str, str]] = {
    "Chanterelle": {
        "english_name": "Chanterelle",
        "swedish_name": "Kantarell",
        "edible": "TRUE",
        "toxicity_level": "SAFE",
    },
    "Fly Agaric": {
        "english_name": "Fly Agaric",
        "swedish_name": "Röd flugsvamp",
        "edible": "FALSE",
        "toxicity_level": "TOXIC",
    },
    "Amanita virosa": {
        "english_name": "Amanita virosa",
        "swedish_name": "Vit flugsvamp",
        "edible": "FALSE",
        "toxicity_level": "EXTREMELY_TOXIC",
    },
    "Porcini": {
        "english_name": "Porcini",
        "swedish_name": "Stensopp",
        "edible": "TRUE",
        "toxicity_level": "SAFE",
    },
}


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_sums_to_one(self):
        scores = {"A": 1.0, "B": 3.0, "C": 2.0}
        result = normalize(scores)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_uniform_when_all_zero(self):
        scores = {"A": 0.0, "B": 0.0, "C": 0.0}
        result = normalize(scores)
        for v in result.values():
            assert abs(v - 1 / 3) < 1e-9

    def test_negatives_clamped_to_zero(self):
        scores = {"A": -5.0, "B": 4.0}
        result = normalize(scores)
        assert result["A"] == 0.0
        assert result["B"] == pytest.approx(1.0)

    def test_single_species(self):
        scores = {"A": 3.5}
        result = normalize(scores)
        assert result["A"] == pytest.approx(1.0)

    def test_preserves_relative_order(self):
        scores = {"A": 1.0, "B": 5.0, "C": 2.0}
        result = normalize(scores)
        assert result["B"] > result["C"] > result["A"]


# ---------------------------------------------------------------------------
# build_prediction
# ---------------------------------------------------------------------------

class TestBuildPrediction:
    def _scores(self):
        return {
            "Fly Agaric": 0.8,
            "Chanterelle": 0.2,
            "Porcini": 0.05,
        }

    def test_top_species_is_highest(self):
        pred = build_prediction("image", "test", self._scores())
        assert pred.species == "Fly Agaric"

    def test_confidence_in_0_1(self):
        pred = build_prediction("trait", "test", self._scores())
        assert 0.0 <= pred.confidence <= 1.0

    def test_method_stored(self):
        pred = build_prediction("llm", "test reason", self._scores())
        assert pred.method == "llm"

    def test_reasoning_stored(self):
        pred = build_prediction("image", "my reasoning", self._scores())
        assert pred.reasoning == "my reasoning"

    def test_top_k_length(self):
        pred = build_prediction("image", "r", self._scores())
        assert len(pred.top_k) <= 5
        assert len(pred.top_k) >= 1


# ---------------------------------------------------------------------------
# trait_scores
# ---------------------------------------------------------------------------

class TestTraitScores:
    def test_red_color_boosts_fly_agaric(self):
        scores = trait_scores({"color": "red"})
        assert scores["Fly Agaric"] > scores["Chanterelle"]
        assert scores["Fly Agaric"] > scores["Porcini"]

    def test_orange_color_boosts_chanterelle(self):
        scores = trait_scores({"color": "orange"})
        assert scores["Chanterelle"] > scores["Fly Agaric"]

    def test_yellow_color_boosts_chanterelle(self):
        scores = trait_scores({"color": "yellow"})
        assert scores["Chanterelle"] > scores["Fly Agaric"]

    def test_brown_color_boosts_porcini(self):
        scores = trait_scores({"color": "brown"})
        assert scores["Porcini"] > scores["Fly Agaric"]

    def test_white_color_boosts_amanita_virosa(self):
        scores = trait_scores({"color": "white"})
        assert scores["Amanita virosa"] > scores["Porcini"]

    def test_decurrent_gills_boost_chanterelle(self):
        scores = trait_scores({"gill_type": "decurrent"})
        assert scores["Chanterelle"] > scores["Fly Agaric"]

    def test_free_gills_boost_fly_agaric(self):
        scores = trait_scores({"gill_type": "free"})
        assert scores["Fly Agaric"] > scores["Chanterelle"]

    def test_ring_stem_boosts_fly_agaric(self):
        scores = trait_scores({"stem_type": "ring/annulus"})
        assert scores["Fly Agaric"] > scores["Porcini"]

    def test_wavy_cap_boosts_chanterelle(self):
        scores = trait_scores({"cap_shape": "wavy"})
        assert scores["Chanterelle"] > scores["Porcini"]

    def test_forest_habitat_boosts_multiple(self):
        scores = trait_scores({"habitat": "forest"})
        assert scores["Fly Agaric"] >= 0.05 + 0.18

    def test_summer_season_boost(self):
        scores_summer = trait_scores({"season": "summer"})
        scores_none = trait_scores({})
        assert scores_summer["Chanterelle"] > scores_none["Chanterelle"]

    def test_returns_all_target_species(self):
        from api.scoring import TARGET_SPECIES
        scores = trait_scores({})
        for sp in TARGET_SPECIES:
            assert sp in scores

    def test_empty_traits_returns_baseline(self):
        scores = trait_scores({})
        for v in scores.values():
            assert v >= 0.0

    def test_case_insensitive(self):
        scores_lower = trait_scores({"color": "red"})
        scores_upper = trait_scores({"color": "RED"})
        assert scores_lower == scores_upper


# ---------------------------------------------------------------------------
# llm_scores
# ---------------------------------------------------------------------------

class TestLlmScores:
    def _zero_metrics(self) -> dict:
        return {
            "red_ratio": 0.0,
            "orange_red_ratio": 0.0,
            "orange_yellow_ratio": 0.0,
            "brown_ratio": 0.0,
            "white_ratio": 0.0,
        }

    def test_red_image_with_white_boosts_fly_agaric(self):
        metrics = {**self._zero_metrics(), "red_ratio": 0.15, "white_ratio": 0.05}
        scores = llm_scores(metrics, {})
        assert scores["Fly Agaric"] > scores["Chanterelle"]

    def test_orange_decurrent_boosts_chanterelle(self):
        scores = llm_scores(self._zero_metrics(), {"color": "orange", "gill_type": "decurrent"})
        assert scores["Chanterelle"] > scores["Fly Agaric"]

    def test_brown_boosts_porcini(self):
        scores = llm_scores(self._zero_metrics(), {"color": "brown"})
        assert scores["Porcini"] > scores["Chanterelle"]

    def test_ring_stem_boosts_fly_agaric(self):
        scores = llm_scores(self._zero_metrics(), {"stem_type": "ring/annulus"})
        assert scores["Fly Agaric"] > scores["Porcini"]

    def test_returns_all_target_species(self):
        from api.scoring import TARGET_SPECIES
        scores = llm_scores(self._zero_metrics(), {})
        for sp in TARGET_SPECIES:
            assert sp in scores

    def test_no_negative_scores(self):
        scores = llm_scores(self._zero_metrics(), {})
        for sp, val in scores.items():
            assert val >= 0, f"Negative score for {sp}"

    def test_orange_red_metric_triggers_fly_agaric(self):
        metrics = {**self._zero_metrics(), "orange_red_ratio": 0.10, "white_ratio": 0.05}
        scores = llm_scores(metrics, {})
        assert scores["Fly Agaric"] > scores["Porcini"]


# ---------------------------------------------------------------------------
# safety_rating
# ---------------------------------------------------------------------------

class TestSafetyRating:
    def test_edible_species(self):
        assert safety_rating("Chanterelle", FAKE_METADATA) == "edible"

    def test_toxic_species(self):
        assert safety_rating("Fly Agaric", FAKE_METADATA) == "inedible"

    def test_extremely_toxic_species(self):
        assert safety_rating("Amanita virosa", FAKE_METADATA) == "inedible"

    def test_unknown_species(self):
        assert safety_rating("Unknown Mushroom", FAKE_METADATA) == "unknown"

    def test_edible_porcini(self):
        assert safety_rating("Porcini", FAKE_METADATA) == "edible"


# ---------------------------------------------------------------------------
# common_name_for
# ---------------------------------------------------------------------------

class TestCommonNameFor:
    def test_returns_english_and_swedish(self):
        en, sv = common_name_for("Chanterelle", FAKE_METADATA)
        assert en == "Chanterelle"
        assert sv == "Kantarell"

    def test_unknown_species_returns_input(self):
        en, sv = common_name_for("Mystery Shroom", FAKE_METADATA)
        assert en == "Mystery Shroom"
        assert sv == "Mystery Shroom"

    def test_fly_agaric_names(self):
        en, sv = common_name_for("Fly Agaric", FAKE_METADATA)
        assert en == "Fly Agaric"
        assert sv == "Röd flugsvamp"


# ---------------------------------------------------------------------------
# adapt_result
# ---------------------------------------------------------------------------

class TestAdaptResult:
    def _hybrid_result(self) -> Dict[str, Any]:
        return {
            "top_species": "Chanterelle",
            "confidence": 0.78,
            "confidence_breakdown": {"image": 0.7, "trait": 0.8, "llm": 0.85},
            "predictions": [
                {"species": "Chanterelle", "confidence": 0.78},
                {"species": "Porcini", "confidence": 0.12},
            ],
            "lookalikes": [
                {"species": "False Chanterelle", "similarity": 0.75, "reason": "Similar colour"},
            ],
            "safety_warnings": [],
            "aggregation_method": "weighted_average",
            "consensus_strength": "strong",
        }

    def _metrics(self) -> dict:
        return {"red_ratio": 0.0, "orange_yellow_ratio": 0.4, "brown_ratio": 0.1, "white_ratio": 0.0}

    def _step1(self) -> dict:
        return {"ml_prediction": {"top_species": "Chanterelle", "confidence": 0.78},
                "visible_traits": {}}

    def test_returns_required_keys(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        for key in ("top_prediction", "overall_confidence", "predictions",
                    "lookalikes", "safety_rating", "trait_extraction", "image_analysis"):
            assert key in result

    def test_top_prediction_matches_hybrid(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        assert result["top_prediction"] == "Chanterelle"

    def test_confidence_value(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        assert result["overall_confidence"] == pytest.approx(0.78)

    def test_lookalike_risk_medium(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        assert result["lookalikes"][0]["risk"] == "medium"

    def test_lookalike_risk_high(self):
        hr = self._hybrid_result()
        hr["lookalikes"][0]["similarity"] = 0.9
        result = adapt_result(hr, self._metrics(), self._step1(), FAKE_METADATA)
        assert result["lookalikes"][0]["risk"] == "high"

    def test_safety_rating_for_edible(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        assert result["safety_rating"] == "edible"

    def test_predictions_have_swedish_name(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        for pred in result["predictions"]:
            assert "swedish_name" in pred

    def test_predictions_have_common_name(self):
        result = adapt_result(self._hybrid_result(), self._metrics(), self._step1(), FAKE_METADATA)
        for pred in result["predictions"]:
            assert "common" in pred
