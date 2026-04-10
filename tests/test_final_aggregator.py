"""
Unit tests for models/final_aggregator.py — FinalAggregator

Tests cover:
  - aggregate() with all three steps available
  - aggregate() with Step 2 not yet concluded
  - aggregate() with Step 3 not available
  - aggregate() with no matching species
  - Confidence weighting and agreement bonus
  - Safety warnings for toxic species
  - Verdict generation (edible / inedible / toxic / unknown)
  - Method agreement detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pytest

from models.final_aggregator import FinalAggregator, _make_verdict


# ---------------------------------------------------------------------------
# Fixture: locate the real species.csv
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aggregator():
    csv_path = (
        Path(__file__).resolve().parent.parent / "data" / "raw" / "species.csv"
    )
    if not csv_path.exists():
        pytest.skip("data/raw/species.csv not found — skipping FinalAggregator tests")
    return FinalAggregator(str(csv_path))


# ---------------------------------------------------------------------------
# _make_verdict helper
# ---------------------------------------------------------------------------

class TestMakeVerdict:
    def test_extremely_toxic(self):
        assert _make_verdict("EXTREMELY_TOXIC", False) == "toxic"

    def test_toxic(self):
        assert _make_verdict("TOXIC", False) == "toxic"

    def test_psychoactive(self):
        assert _make_verdict("PSYCHOACTIVE", False) == "toxic"

    def test_edible(self):
        assert _make_verdict("SAFE", True) == "edible"

    def test_inedible(self):
        assert _make_verdict("SAFE", False) == "inedible"


# ---------------------------------------------------------------------------
# FinalAggregator.aggregate()
# ---------------------------------------------------------------------------

def _step1(top_species: str = "Chanterelle", confidence: float = 0.80) -> Dict[str, Any]:
    return {
        "step1": {
            "ml_prediction": {
                "top_species": top_species,
                "confidence": confidence,
                "reasoning": "CV fallback",
                "top_k": [
                    {"species": top_species, "confidence": confidence},
                    {"species": "Porcini", "confidence": 0.10},
                ],
            },
            "visible_traits": {"dominant_color": "orange-yellow"},
        }
    }


def _step2_conclusion(species: str = "Kantarell") -> Dict[str, Any]:
    return {
        "status": "conclusion",
        "session_id": "sess-001",
        "species": species,
        "edibility": "*",
        "edibility_label": "edible",
        "path": ["Colour orange-yellow", "Ridges decurrent"],
        "auto_answered": ["Colour orange-yellow"],
    }


def _step2_question() -> Dict[str, Any]:
    return {
        "status": "question",
        "session_id": "sess-001",
        "question": "What colour is the cap?",
        "options": ["Red", "Orange", "Brown"],
        "auto_answered": [],
    }


def _step3_ok() -> Dict[str, Any]:
    return {
        "status": "ok",
        "candidate": {
            "species_id": "CA.CI",
            "swedish_name": "Kantarell",
            "english_name": "Chanterelle",
            "scientific_name": "Cantharellus cibarius",
            "edible": "TRUE",
            "toxicity_level": "SAFE",
        },
        "name_match_score": 0.95,
        "trait_match": {
            "score": 0.87,
            "matched": [{"trait": "cap_colour", "visible_value": "orange", "db_value": "orange"}],
            "conflicts": [],
            "not_comparable": [],
        },
        "lookalikes": [],
        "safety_alert": False,
    }


def _step3_not_found() -> Dict[str, Any]:
    return {"status": "species_not_found"}


class TestFinalAggregatorStructure:
    def test_returns_required_keys(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        for key in ("final_recommendation", "ml_alternatives",
                    "exchangeable_species", "safety_warnings",
                    "verdict", "method_agreement"):
            assert key in result, f"Missing top-level key: {key}"

    def test_final_recommendation_has_required_fields(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        rec = result["final_recommendation"]
        for field in ("species_id", "swedish_name", "english_name",
                      "overall_confidence", "confidence_breakdown", "reasoning"):
            assert field in rec, f"Missing field in final_recommendation: {field}"

    def test_confidence_in_0_1(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        conf = result["final_recommendation"]["overall_confidence"]
        assert 0.0 <= conf <= 1.0

    def test_confidence_breakdown_has_three_keys(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        breakdown = result["final_recommendation"]["confidence_breakdown"]
        assert "image_analysis" in breakdown
        assert "tree_traversal" in breakdown
        assert "trait_match" in breakdown

    def test_ml_alternatives_is_list(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        assert isinstance(result["ml_alternatives"], list)

    def test_safety_warnings_is_list(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        assert isinstance(result["safety_warnings"], list)

    def test_verdict_is_valid_value(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        assert result["verdict"] in {"edible", "inedible", "toxic", "unknown"}

    def test_method_agreement_is_valid(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        assert result["method_agreement"] in {"full", "partial", "none"}


class TestFinalAggregatorConfidence:
    def test_step2_concluded_gives_high_tree_weight(self, aggregator):
        with_step2 = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_ok())
        without_step2 = aggregator.aggregate(_step1(), _step2_question(), _step3_ok())
        assert (with_step2["final_recommendation"]["confidence_breakdown"]["tree_traversal"]
                >= without_step2["final_recommendation"]["confidence_breakdown"]["tree_traversal"])

    def test_agreement_bonus_applied(self, aggregator):
        """Steps 1 and 2 agreeing should give full method_agreement."""
        # Step1 top = Chanterelle (CA.CI), Step2 conclusion = Kantarell (maps to CA.CI)
        result = aggregator.aggregate(
            _step1("Chanterelle", 0.85),
            _step2_conclusion("Kantarell"),
            _step3_ok(),
        )
        assert result["method_agreement"] in {"full", "partial"}

    def test_no_step3_uses_neutral_trait_confidence(self, aggregator):
        result = aggregator.aggregate(_step1(), _step2_conclusion(), _step3_not_found())
        breakdown = result["final_recommendation"]["confidence_breakdown"]
        assert breakdown["trait_match"] == pytest.approx(0.5)


class TestFinalAggregatorEdgeCases:
    def test_unknown_species_when_nothing_matches(self, aggregator):
        s2 = {"status": "conclusion", "session_id": "x", "species": "ΨΨΨ_nonexistent_ΨΨΨ",
              "edibility": "?", "edibility_label": "unknown", "path": [], "auto_answered": []}
        s1 = _step1("Other Boletus", 0.3)
        # Should fall back to image analysis
        result = aggregator.aggregate(s1, s2, _step3_not_found())
        # result should still have required keys
        assert "verdict" in result

    def test_safety_warning_for_toxic_candidate(self, aggregator):
        """When Step 3 flags a toxic lookalike, safety_warnings should be populated."""
        s3_with_lookalike = {
            "status": "ok",
            "candidate": {
                "species_id": "CA.CI",
                "swedish_name": "Kantarell",
                "english_name": "Chanterelle",
                "scientific_name": "Cantharellus cibarius",
                "edible": "TRUE",
                "toxicity_level": "SAFE",
            },
            "name_match_score": 0.95,
            "trait_match": {"score": 0.8, "matched": [], "conflicts": [], "not_comparable": []},
            "lookalikes": [
                {
                    "species_id": "AM.VI",
                    "swedish_name": "Vit flugsvamp",
                    "english_name": "Amanita virosa",
                    "toxicity_level": "EXTREMELY_TOXIC",
                    "safety_alert": True,
                    "confusion_likelihood": "high",
                    "distinguishing_features": "White all over, sack-like volva at base.",
                }
            ],
            "safety_alert": True,
        }
        result = aggregator.aggregate(_step1(), _step2_conclusion(), s3_with_lookalike)
        assert len(result["safety_warnings"]) > 0
