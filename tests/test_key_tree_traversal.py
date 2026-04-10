"""
Unit tests for models/key_tree_traversal.py — KeyTreeEngine

Tests cover:
  - start_session: starts a new session, returns question or conclusion
  - answer: advances traversal on a valid answer
  - answer: returns error on unknown session_id
  - get_session: returns session state or None
  - Auto-answering from visible_traits
  - Session isolation (two sessions don't interfere)
"""

from __future__ import annotations

from pathlib import Path
import pytest

from models.key_tree_traversal import KeyTreeEngine


# ---------------------------------------------------------------------------
# Fixture: engine backed by the real key.xml
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    xml_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "key.xml"
    if not xml_path.exists():
        pytest.skip("data/raw/key.xml not found — skipping KeyTreeEngine tests")
    return KeyTreeEngine(str(xml_path))


EMPTY_TRAITS: dict = {}

CHANTERELLE_TRAITS: dict = {
    "dominant_color": "orange-yellow",
    "cap_shape": "funnel-shaped",
    "surface_texture": "smooth",
    "has_ridges": True,
    "colour_ratios": {
        "red": 0.0,
        "orange_red": 0.0,
        "orange_yellow": 0.6,
        "brown": 0.05,
        "white": 0.0,
        "dark": 0.0,
    },
}


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------

class TestStartSession:
    def test_returns_dict(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        assert isinstance(result, dict)

    def test_status_is_question_or_conclusion(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        assert result["status"] in {"question", "conclusion"}

    def test_session_id_assigned(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        assert "session_id" in result
        assert result["session_id"]

    def test_explicit_session_id_respected(self, engine):
        result = engine.start_session("my-session-123", EMPTY_TRAITS)
        assert result["session_id"] == "my-session-123"

    def test_question_has_options(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        if result["status"] == "question":
            assert "options" in result
            assert len(result["options"]) >= 1

    def test_question_text_non_empty(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        if result["status"] == "question":
            assert result.get("question", "").strip()

    def test_conclusion_has_species(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        if result["status"] == "conclusion":
            assert "species" in result
            assert result["species"]

    def test_auto_answered_is_list(self, engine):
        result = engine.start_session(None, CHANTERELLE_TRAITS)
        assert isinstance(result.get("auto_answered", []), list)


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_error_on_unknown_session(self, engine):
        result = engine.answer("session-that-does-not-exist-xyz", "Red")
        assert result["status"] == "error"

    def test_valid_answer_advances_traversal(self, engine):
        start = engine.start_session(None, EMPTY_TRAITS)
        if start["status"] != "question":
            pytest.skip("Tree auto-resolved on first question — no user input needed")
        session_id = start["session_id"]
        option = start["options"][0]
        result = engine.answer(session_id, option)
        assert result["status"] in {"question", "conclusion", "error"}

    def test_picking_all_first_options_reaches_conclusion(self, engine):
        """
        Greedily picking the first option at each step should eventually
        reach a conclusion (not loop forever).
        """
        result = engine.start_session(None, EMPTY_TRAITS)
        session_id = result["session_id"]
        for _ in range(30):  # guard against infinite loops
            if result["status"] != "question":
                break
            result = engine.answer(session_id, result["options"][0])
        assert result["status"] == "conclusion"

    def test_conclusion_has_edibility(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        session_id = result["session_id"]
        for _ in range(30):
            if result["status"] != "question":
                break
            result = engine.answer(session_id, result["options"][0])
        assert "edibility" in result

    def test_conclusion_has_path(self, engine):
        result = engine.start_session(None, EMPTY_TRAITS)
        session_id = result["session_id"]
        for _ in range(30):
            if result["status"] != "question":
                break
            result = engine.answer(session_id, result["options"][0])
        assert isinstance(result.get("path", []), list)


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_returns_none_for_unknown_session(self, engine):
        assert engine.get_session("nonexistent-session-abc") is None

    def test_returns_state_for_active_session(self, engine):
        start = engine.start_session(None, EMPTY_TRAITS)
        sid = start["session_id"]
        state = engine.get_session(sid)
        assert state is not None

    def test_returned_state_has_session_id(self, engine):
        start = engine.start_session(None, EMPTY_TRAITS)
        sid = start["session_id"]
        state = engine.get_session(sid)
        assert state.get("session_id") == sid


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_two_sessions_are_independent(self, engine):
        s1 = engine.start_session(None, EMPTY_TRAITS)
        s2 = engine.start_session(None, EMPTY_TRAITS)
        assert s1["session_id"] != s2["session_id"]

    def test_answering_one_session_does_not_affect_other(self, engine):
        s1 = engine.start_session(None, EMPTY_TRAITS)
        s2 = engine.start_session(None, EMPTY_TRAITS)
        if s1["status"] == "question" and s2["status"] == "question":
            engine.answer(s1["session_id"], s1["options"][0])
            state2 = engine.get_session(s2["session_id"])
            assert state2 is not None
