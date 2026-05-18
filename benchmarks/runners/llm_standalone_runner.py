"""Standalone vision-LLM benchmark runner.

Feeds both above and below images directly to the LLM with a minimal
mycological prompt — no trait extraction, no tree, no CNN, no database.
This establishes a pure "raw vision" baseline for System A.
"""

import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from benchmarks.runners.base import RunnerResult
from benchmarks.runners.unified_runner import resolve_species_name

logger = logging.getLogger(__name__)


def _image_to_b64(image_bytes: bytes, max_side: int = 256, quality: int = 80) -> str:
    """Downsample an image before sending to the local vision LLM."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    pil.thumbnail((max_side, max_side), resampling)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Extract species name and confidence from LLM response text."""
    # Try direct JSON parse
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"^```json\s*", "", response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try JSON again
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find largest JSON object
    matches = re.findall(r"\{.*?\}", cleaned, re.DOTALL)
    if matches:
        largest = max(matches, key=len)
        try:
            data = json.loads(largest)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: regex for species name
    species_match = re.search(
        r'(?:species|name)["\']?\s*[:=]\s*["\']?([^"\'\n,]+)',
        response,
        re.IGNORECASE,
    )
    species = species_match.group(1).strip() if species_match else "Unknown"

    conf_match = re.search(
        r'(?:confidence|conf)["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)',
        response,
        re.IGNORECASE,
    )
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    return {
        "species": species,
        "confidence": confidence,
        "reasoning": response,
    }


SYSTEM_PROMPT = """You are an expert mycologist. Identify the mushroom species from the two provided images.
One image shows the cap/top view, the other shows the underside/gills view.

Even if you are uncertain, always provide your best guess. Do not return "Unknown".
Use the confidence score to indicate your level of certainty:
  0.0–0.3 = low confidence (educated guess)
  0.3–0.6 = moderate confidence (plausible match)
  0.6–0.9 = high confidence (strong visual match)
  0.9–1.0 = very high confidence (diagnostic features present)

Return your answer as JSON:
{
    "species": "English common name",
    "confidence": 0.45,
    "reasoning": "Brief explanation of key visual features and why you chose this species"
}

Be precise and prioritize safety. If the mushroom might be toxic, mention this in your reasoning."""


class LLMStandaloneRunner:
    """Raw vision-LLM baseline: images → LLM → species prediction."""

    name = "llm"

    def __init__(self, backend=None):
        from models.llm_classifier import OllamaBackend, LLMClassifier

        if backend is not None:
            self.backend = backend
        elif OllamaBackend.is_available():
            self.backend = LLMClassifier(backend_type="ollama")
        else:
            self.backend = None

    def predict(self, specimen) -> RunnerResult:
        """Run standalone vision LLM on above + below photos.

        Args:
            specimen: BenchmarkSpecimen with ``above_path`` and ``below_path``.

        Returns:
            RunnerResult with the LLM's top prediction.
        """
        above = specimen.load_above_bytes()
        below = specimen.load_below_bytes()

        if not above or not below:
            return RunnerResult(
                method_name="llm",
                predictions=[],
                coverage=False,
                error="Missing above or below photo",
            )

        if self.backend is None:
            return RunnerResult(
                method_name="llm",
                predictions=[],
                coverage=False,
                error="LLM backend not available",
            )

        images_b64 = [_image_to_b64(above), _image_to_b64(below)]

        t0 = time.perf_counter()
        try:
            response = self.backend.backend.query(
                system_prompt=SYSTEM_PROMPT,
                user_observation="Identify the mushroom species from these two images.",
                images=images_b64,
            )
        except Exception as exc:
            return RunnerResult(
                method_name="llm",
                predictions=[],
                coverage=False,
                error=f"LLM query failed: {exc}",
                inference_time_ms=(time.perf_counter() - t0) * 1000,
            )
        elapsed = (time.perf_counter() - t0) * 1000

        parsed = _parse_llm_response(response)
        species_raw = parsed.get("species", "Unknown")
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", "")

        if str(species_raw).lower() in ("unknown", "error", "none", "n/a", ""):
            return RunnerResult(
                method_name="llm",
                predictions=[],
                coverage=False,
                error="LLM could not identify the species",
                inference_time_ms=elapsed,
                metadata={"llm_reasoning": reasoning, "raw_response": response},
            )

        species_id = resolve_species_name(species_raw)
        top_label = species_id if species_id else species_raw

        return RunnerResult(
            method_name="llm",
            predictions=[(top_label, confidence)] if top_label else [],
            coverage=True,
            inference_time_ms=elapsed,
            metadata={
                "llm_reasoning": reasoning,
                "raw_response": response,
                "resolved_species_id": species_id,
            },
        )
