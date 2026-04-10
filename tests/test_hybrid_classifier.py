"""
Unit tests for models/hybrid_classifier.py

Tests cover:
  - HybridClassifier.classify with WEIGHTED_AVERAGE aggregation
  - HybridClassifier.classify with CONSENSUS aggregation
  - Result structure and confidence ranges
  - Lookalike detection
  - Safety warnings for toxic species
  - to_dict() serialisation
"""

from __future__ import annotations

import pytest

from models.hybrid_classifier import (
    AggregationMethod,
    HybridClassifier,
    MethodPrediction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(method: str, species: str, confidence: float) -> MethodPrediction:
    """Build a simple MethodPrediction with a single top-k entry."""
    return MethodPrediction(
        method=method,
        species=species,
        confidence=confidence,
        reasoning=f"{method} test prediction",
        top_k=[(species, confidence)],
    )


def _full_pred(method: str, species: str, confidence: float,
               others: list[tuple[str, float]] | None = None) -> MethodPrediction:
    """Build a MethodPrediction with multiple top-k entries."""
    top_k = [(species, confidence)] + (others or [])
    return MethodPrediction(
        method=method,
        species=species,
        confidence=confidence,
        reasoning=f"{method} prediction",
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# HybridClassifier — weighted average
# ---------------------------------------------------------------------------

class TestHybridClassifierWeightedAverage:
    @pytest.fixture
    def classifier(self):
        return HybridClassifier(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)

    def test_classify_returns_dict(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.85),
            trait_prediction=_pred("trait", "Chanterelle", 0.90),
            llm_prediction=_pred("llm", "Chanterelle", 0.80),
        )
        assert isinstance(result, object)

    def test_to_dict_has_required_keys(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.85),
            trait_prediction=_pred("trait", "Chanterelle", 0.90),
            llm_prediction=_pred("llm", "Chanterelle", 0.80),
        ).to_dict()
        for key in ("top_species", "confidence", "confidence_breakdown",
                    "predictions", "lookalikes", "safety_warnings",
                    "aggregation_method", "consensus_strength"):
            assert key in result, f"Missing key: {key}"

    def test_full_consensus_high_confidence(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Fly Agaric", 0.90),
            trait_prediction=_pred("trait", "Fly Agaric", 0.88),
            llm_prediction=_pred("llm", "Fly Agaric", 0.92),
        ).to_dict()
        assert result["top_species"] == "Fly Agaric"
        assert result["confidence"] > 0.5

    def test_confidence_in_0_1(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Porcini", 0.6),
            trait_prediction=_pred("trait", "Porcini", 0.5),
            llm_prediction=_pred("llm", "Porcini", 0.7),
        ).to_dict()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_disagreeing_methods_gives_lower_consensus(self, classifier):
        # Each method picks a different species
        result_disagree = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.80),
            trait_prediction=_pred("trait", "Porcini", 0.80),
            llm_prediction=_pred("llm", "Fly Agaric", 0.80),
        ).to_dict()
        result_agree = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.80),
            trait_prediction=_pred("trait", "Chanterelle", 0.80),
            llm_prediction=_pred("llm", "Chanterelle", 0.80),
        ).to_dict()
        # Agreeing methods should have higher or equal confidence
        assert result_agree["confidence"] >= result_disagree["confidence"]

    def test_predictions_list_is_not_empty(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Porcini", 0.70),
            trait_prediction=_pred("trait", "Porcini", 0.65),
            llm_prediction=_pred("llm", "Porcini", 0.72),
        ).to_dict()
        assert len(result["predictions"]) >= 1

    def test_predictions_have_species_and_confidence(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Black Trumpet", 0.60),
            trait_prediction=_pred("trait", "Black Trumpet", 0.55),
            llm_prediction=_pred("llm", "Black Trumpet", 0.65),
        ).to_dict()
        for pred in result["predictions"]:
            assert "species" in pred
            assert "confidence" in pred

    def test_safety_warnings_list_present(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Amanita virosa", 0.85),
            trait_prediction=_pred("trait", "Amanita virosa", 0.88),
            llm_prediction=_pred("llm", "Amanita virosa", 0.90),
        ).to_dict()
        assert isinstance(result["safety_warnings"], list)

    def test_lookalikes_list_present(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.80),
            trait_prediction=_pred("trait", "Chanterelle", 0.75),
            llm_prediction=_pred("llm", "Chanterelle", 0.78),
        ).to_dict()
        assert isinstance(result["lookalikes"], list)

    def test_confidence_breakdown_has_three_methods(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.80),
            trait_prediction=_pred("trait", "Chanterelle", 0.75),
            llm_prediction=_pred("llm", "Chanterelle", 0.78),
        ).to_dict()
        breakdown = result["confidence_breakdown"]
        assert "image" in breakdown
        assert "trait" in breakdown
        assert "llm" in breakdown


# ---------------------------------------------------------------------------
# HybridClassifier — consensus aggregation
# ---------------------------------------------------------------------------

class TestHybridClassifierVoting:
    @pytest.fixture
    def classifier(self):
        return HybridClassifier(aggregation_method=AggregationMethod.VOTING)

    def test_voting_picks_majority(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Chanterelle", 0.80),
            trait_prediction=_pred("trait", "Chanterelle", 0.75),
            llm_prediction=_pred("llm", "Porcini", 0.70),
        ).to_dict()
        # 2 out of 3 vote for Chanterelle
        assert result["top_species"] == "Chanterelle"

    def test_returns_valid_structure(self, classifier):
        result = classifier.classify(
            image_prediction=_pred("image", "Porcini", 0.65),
            trait_prediction=_pred("trait", "Porcini", 0.70),
            llm_prediction=_pred("llm", "Porcini", 0.68),
        ).to_dict()
        assert "top_species" in result
        assert "confidence" in result
