"""Benchmark runner for the Swedish decision-tree classifier.

The comparative benchmark uses automatic visual-trait answers only. If the
tree cannot answer the next question from extracted traits, it abstains.
"""

import time

from benchmarks.config import KEY_XML
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from benchmarks.runners._trait_helper import get_merged_extracted_traits
from models.key_tree_traversal import KeyTreeEngine
from models.cnn_classifier import get_classifier


class TreeRunner(BenchmarkRunner):
    """Wrapper around ``models.key_tree_traversal.KeyTreeEngine``.

    Supports two trait sources:
      - "extracted" (a1_tree): merged computer-vision traits
      - "oracle"    (a2_tree): ground-truth traits from SpeciesTraitOracle
    """

    name = "tree"

    def __init__(self, trait_source: str = "extracted", oracle_trait_provider=None):
        if trait_source not in ("extracted", "oracle"):
            raise ValueError(f"TreeRunner trait_source must be 'extracted' or 'oracle', got {trait_source}")
        self.engine = KeyTreeEngine(str(KEY_XML))
        self.trait_source = trait_source
        self.oracle_trait_provider = oracle_trait_provider

    def predict(self, specimen) -> RunnerResult:
        """Traverse the decision tree for a single specimen.

        Args:
            specimen: BenchmarkSpecimen with ``above_path``, ``below_path``,
                and ``species_id``.

        Returns:
            ``RunnerResult`` with a single prediction when the tree reaches
            a conclusion, or ``coverage=False`` when it gets stuck.
        """
        t0 = time.perf_counter()

        # ---- Obtain traits ----
        if self.trait_source == "extracted":
            visible_traits = get_merged_extracted_traits(specimen)
        elif self.trait_source == "oracle" and self.oracle_trait_provider is not None:
            # Use oracle extractor-shaped output (same dict structure as CV extractor)
            # Need case hint; default to "classical" if not determinable
            visible_traits = self.oracle_trait_provider.get_extractor_output(
                specimen.species_id, case="classical"
            )
        else:
            visible_traits = {}

        # ---- CNN hint (optional, same as original TreeRunner) ----
        ml_hint = None
        try:
            cnn = get_classifier()
            if cnn.is_trained:
                above = specimen.load_above_bytes()
                below = specimen.load_below_bytes()
                photo = above or below
                if photo:
                    cnn_scores = cnn.predict(photo)
                    if cnn_scores is not None:
                        ordered = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
                        top_species, top_conf = ordered[0]
                        ml_hint = {
                            "top_species": top_species,
                            "confidence": round(top_conf, 4),
                        }
        except Exception:
            pass

        result = self.engine.start_session(None, visible_traits, ml_hint, pre_answers=None)
        session_id = result.get("session_id")

        # Clean up session if it wasn't already deleted by answer().
        if session_id and session_id in self.engine._sessions:
            del self.engine._sessions[session_id]

        elapsed = (time.perf_counter() - t0) * 1000

        if result.get("status") == "conclusion":
            swedish_name = result["species"]
            species_id = self._resolve_swedish_name(swedish_name)
            return RunnerResult(
                method_name=f"tree_{self.trait_source}",
                predictions=[(species_id, 1.0)],
                coverage=True,
                inference_time_ms=elapsed,
                metadata={
                    "swedish_name": swedish_name,
                    "auto_answered": result.get("auto_answered", []),
                    "path": result.get("path", []),
                    "trait_source": self.trait_source,
                },
            )

        # Tree got stuck on a question it could not answer.
        return RunnerResult(
            method_name=f"tree_{self.trait_source}",
            predictions=[],
            coverage=False,
            inference_time_ms=elapsed,
            metadata={
                "stuck_at_question": result.get("question"),
                "trait_source": self.trait_source,
            },
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
