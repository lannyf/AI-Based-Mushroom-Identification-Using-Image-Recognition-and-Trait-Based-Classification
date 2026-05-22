"""Benchmark runner for the unified identification pipeline.

Wraps ``models.unified_pipeline.UnifiedPipeline`` and produces a standard
``RunnerResult`` so the comparative benchmark engine can treat it uniformly
alongside CNN, tree, and trait-database runners.
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional

from benchmarks.config import SPECIES_CSV, KEY_XML
from benchmarks.runners.base import RunnerResult


def _build_name_resolver() -> Dict[str, str]:
    """Build a comprehensive name → species_id lookup.

    Covers English names, Swedish names, scientific names, and the existing
    CNN output-name mappings so that LLM responses can be normalised reliably.
    """
    from benchmarks.config import CNN_NAME_TO_SPECIES_ID

    resolver: Dict[str, str] = dict(CNN_NAME_TO_SPECIES_ID)
    with open(SPECIES_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = row["species_id"]
            for key in (row["english_name"], row["swedish_name"], row["scientific_name"]):
                if key:
                    resolver[key.lower().strip()] = sid
                    # Strip common suffixes that LLMs may drop
                    resolver[key.lower().strip().replace(" mushroom", "")] = sid
    return resolver


_NAME_RESOLVER = _build_name_resolver()


def resolve_species_name(name: str) -> Optional[str]:
    """Map a free-text species name (English/Swedish/scientific) to a canonical species_id.

    Returns ``None`` when the name cannot be resolved or is clearly a failure state
    ("Unknown", "Error", empty string, etc.).
    """
    if not name:
        return None
    name_lower = name.lower().strip()
    if name_lower in ("unknown", "error", "unable to parse", "none", "n/a"):
        return None

    # Exact match
    if name_lower in _NAME_RESOLVER:
        return _NAME_RESOLVER[name_lower]

    # Substring containment (both directions)
    for key, sid in _NAME_RESOLVER.items():
        if len(key) > 3 and (name_lower in key or key in name_lower):
            return sid

    return None


class UnifiedRunner:
    """Benchmark wrapper around the full unified pipeline (YOLO → traits → CNN → tree → DB → LLM).

    Supports two modes:
      - B1 (no oracle):  uses real trait extractor output
      - B2 (oracle):     uses perfect oracle traits from species_traits.xml
    """

    name = "unified_b1"

    def __init__(self, segmenter=None, llm_backend=None, oracle_trait_provider=None):
        from models.unified_pipeline import UnifiedPipeline
        from models.mushroom_segmenter import get_segmenter
        from models.llm_classifier import OllamaBackend, LLMClassifier
        from models.key_tree_parser import KeyTreeParser
        from benchmarks.config import YOLO_WEIGHTS

        # Lazy-load segmenter if not provided
        if segmenter is None:
            if YOLO_WEIGHTS.exists():
                try:
                    segmenter = get_segmenter(model_path=str(YOLO_WEIGHTS))
                except Exception as exc:
                    segmenter = None
            else:
                segmenter = None

        # Parse key.xml for LLM injection
        key_text = ""
        try:
            ktp = KeyTreeParser(str(KEY_XML))
            key_text = ktp.get_prompt_injection()
        except Exception:
            pass

        # Lazy-init LLM if not provided
        if llm_backend is None:
            if OllamaBackend.is_available():
                try:
                    llm_backend = LLMClassifier(backend_type="ollama", key_tree_text=key_text)
                except Exception:
                    llm_backend = None

        self.pipeline = UnifiedPipeline(
            segmenter=segmenter,
            key_xml_path=str(KEY_XML),
            llm_backend=llm_backend,
            auto_init_llm=(llm_backend is None),
        )
        self.oracle_trait_provider = oracle_trait_provider

    def predict(self, specimen) -> RunnerResult:
        """Run the unified pipeline on a specimen with above + below photos.

        Args:
            specimen: ``BenchmarkSpecimen`` (needs both ``above_path`` and
                ``below_path`` to be present).

        Returns:
            ``RunnerResult`` with the LLM's top prediction mapped to a
            canonical ``species_id`` where possible.
        """
        above = specimen.load_above_bytes()
        below = specimen.load_below_bytes()

        if not above or not below:
            return RunnerResult(
                method_name="unified",
                predictions=[],
                coverage=False,
                error="Missing above or below photo for unified pipeline",
            )

        t0 = time.perf_counter()
        try:
            result = self.pipeline.run(
                above, below,
                species_id=specimen.species_id,
                oracle_trait_provider=self.oracle_trait_provider,
            )
        except Exception as exc:
            return RunnerResult(
                method_name="unified",
                predictions=[],
                coverage=False,
                error=f"Pipeline exception: {exc}",
                inference_time_ms=(time.perf_counter() - t0) * 1000,
            )
        elapsed = (time.perf_counter() - t0) * 1000

        llm_dict = result.get("llm", {})
        top_species_raw = llm_dict.get("top_species", "Unknown")
        top_conf = float(llm_dict.get("confidence", 0.0))

        if str(top_species_raw).lower() in ("unknown", "error", "unable to parse", "none", "n/a"):
            return RunnerResult(
                method_name="unified",
                predictions=[],
                coverage=False,
                error=llm_dict.get("reasoning", "Unified LLM produced no usable prediction"),
                inference_time_ms=elapsed,
                metadata={
                    "agreement": result.get("agreement", "inconclusive"),
                    "case": result.get("case", {}).get("case", "unknown"),
                    "needs_clarification": llm_dict.get("needs_clarification", False),
                    "llm_reasoning": llm_dict.get("reasoning", ""),
                    "oracle_used": self.oracle_trait_provider is not None,
                },
            )

        species_id = resolve_species_name(top_species_raw)
        top_label = species_id if species_id else top_species_raw

        predictions = [(top_label, top_conf)] if top_label else []

        # Pull intermediate signals for metadata / agreement analysis
        cnn_dict = result.get("cnn", {})
        tree_dict = result.get("tree", {})
        db_dict = result.get("database", {})

        cnn_pred = cnn_dict.get("species")
        tree_pred = (
            tree_dict.get("species")
            if tree_dict.get("status") == "conclusion"
            else None
        )
        db_candidate = db_dict.get("candidate", {})
        db_pred = db_candidate.get("english_name")
        db_swedish = db_candidate.get("swedish_name")

        # Resolve DB prediction to species_id too
        db_species_id = resolve_species_name(db_pred) if db_pred else None
        if not db_species_id and db_swedish:
            db_species_id = resolve_species_name(db_swedish)

        return RunnerResult(
            method_name="unified",
            predictions=predictions,
            coverage=True,
            inference_time_ms=elapsed,
            metadata={
                "agreement": result.get("agreement", "inconclusive"),
                "case": result.get("case", {}).get("case", "unknown"),
                "case_confidence": result.get("case", {}).get("confidence", 0.0),
                "llm_reasoning": llm_dict.get("reasoning", ""),
                "needs_clarification": llm_dict.get("needs_clarification", False),
                "cnn_pred": cnn_pred,
                "cnn_conclusive": cnn_dict.get("conclusive", False),
                "cnn_confidence": cnn_dict.get("confidence", 0.0),
                "tree_pred": tree_pred,
                "db_pred": db_pred,
                "db_species_id": db_species_id,
                "db_score": db_dict.get("trait_match", {}).get("score", 0.0),
                "final_recommendation": result.get("final_recommendation", {}),
                "oracle_used": self.oracle_trait_provider is not None,
            },
        )
