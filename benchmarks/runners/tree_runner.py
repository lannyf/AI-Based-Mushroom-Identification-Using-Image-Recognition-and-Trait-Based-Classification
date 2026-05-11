"""Benchmark runner for the Swedish decision-tree classifier.

Supports two modes:

* ``auto`` — answers questions automatically from visual traits; stops
  at the first question it cannot resolve.
* ``oracle`` — feeds pre-recorded ground-truth answers from
  ``oracle_answers.json`` so the tree is evaluated under ideal
  conditions.
"""

import json
import time
from pathlib import Path
from typing import Dict

from benchmarks.config import KEY_XML, ORACLE_JSON
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from models.key_tree_traversal import KeyTreeEngine
from benchmarks.runners._extract_cache import extract
from models.cnn_classifier import get_classifier


class TreeRunner(BenchmarkRunner):
    """Wrapper around ``models.key_tree_traversal.KeyTreeEngine``.

    The tree outputs Swedish names; ``_resolve_swedish_name()`` maps
    those to canonical ``species_id`` values via the XML aliases table
    and ``species.csv`` fallback.
    """

    name = "tree"

    def __init__(self, mode: str = "auto"):
        self.engine = KeyTreeEngine(str(KEY_XML))
        self.mode = mode
        self.oracle: Dict[str, Dict[str, str]] = {}
        if mode == "oracle" and Path(ORACLE_JSON).exists():
            with open(ORACLE_JSON, encoding="utf-8") as f:
                self.oracle = json.load(f)

    def predict(self, sample) -> RunnerResult:
        """Traverse the decision tree for a single sample.

        In ``auto`` mode the loop breaks on the first unanswered question.
        In ``oracle`` mode it continues until a conclusion is reached or
        the oracle has no more answers.

        Args:
            sample: ``BenchmarkSample`` with ``image_bytes`` and ``species_id``.

        Returns:
            ``RunnerResult`` with a single prediction when the tree reaches
            a conclusion, or ``coverage=False`` when it gets stuck.
        """
        t0 = time.perf_counter()

        step1_result = extract(sample.image_bytes)
        visible_traits = step1_result["visible_traits"]

        ml_hint = None
        try:
            cnn = get_classifier()
            if cnn.is_trained:
                cnn_scores = cnn.predict(sample.image_bytes)
                if cnn_scores is not None:
                    ordered = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
                    top_species, top_conf = ordered[0]
                    ml_hint = {
                        "top_species": top_species,
                        "confidence": round(top_conf, 4),
                    }
        except Exception:
            pass

        result = self.engine.start_session(None, visible_traits, ml_hint)
        session_id = result.get("session_id")

        # Answer questions until we hit a conclusion or run out of answers.
        while result.get("status") == "question":
            if self.mode == "oracle":
                species_answers = self.oracle.get(sample.species_id, {})
                answer = species_answers.get(result["question"])
                if answer is None:
                    break
                result = self.engine.answer(session_id, answer)
            else:
                break

        # Clean up session if it wasn't already deleted by answer().
        if session_id and session_id in self.engine._sessions:
            del self.engine._sessions[session_id]

        elapsed = (time.perf_counter() - t0) * 1000

        if result.get("status") == "conclusion":
            swedish_name = result["species"]
            species_id = self._resolve_swedish_name(swedish_name)
            return RunnerResult(
                method_name=f"tree_{self.mode}",
                predictions=[(species_id, 1.0)],
                coverage=True,
                inference_time_ms=elapsed,
                metadata={
                    "swedish_name": swedish_name,
                    "auto_answered": result.get("auto_answered", []),
                    "path": result.get("path", []),
                },
            )

        # Tree got stuck on a question it could not answer.
        return RunnerResult(
            method_name=f"tree_{self.mode}",
            predictions=[],
            coverage=False,
            inference_time_ms=elapsed,
            metadata={"stuck_at_question": result.get("question")},
        )

    def _resolve_swedish_name(self, swedish_name: str) -> str:
        """Map a Swedish tree output name to a canonical ``species_id``.

        First tries the hard-coded alias table from the XML parser, then
        falls back to scanning ``species.csv`` by Swedish or English name.
        """
        from models.trait_database_comparator import _KEY_XML_ALIASES

        alias = _KEY_XML_ALIASES.get(swedish_name.lower().strip())
        if alias:
            return alias

        # Fallback: lookup in species.csv by swedish_name or english_name
        import csv
        from benchmarks.config import SPECIES_CSV

        with open(SPECIES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["swedish_name"].lower().strip() == swedish_name.lower().strip():
                    return row["species_id"]
                if row["english_name"].lower().strip() == swedish_name.lower().strip():
                    return row["species_id"]
        return swedish_name
