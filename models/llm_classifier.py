"""
LLM-Based Mushroom Classifier

Implements natural language processing for mushroom identification using
local Large Language Models via Ollama. Supports a mock backend for testing
when Ollama is not available.

The module provides:
1. LLMPromptTemplate: System prompts with mushroom expertise context
2. LLMClassifier: API client and response parsing
3. SpeciesDatabase: In-memory species lookup
4. PredictionResult: Standardized output format
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Standard prediction output format compatible with Phase 3/4 methods."""
    
    top_species: str
    top_confidence: float
    predictions: List[Tuple[str, float, str]]
    reasoning: str
    safety_warnings: List[str]
    model_used: str
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'top_species': self.top_species,
            'confidence': self.top_confidence,
            'predictions': [
                {'species': p[0], 'confidence': p[1], 'reason': p[2]}
                for p in self.predictions
            ],
            'reasoning': self.reasoning,
            'safety_warnings': self.safety_warnings,
            'model_used': self.model_used,
            'processing_time_ms': self.processing_time_ms
        }


class SpeciesDatabase:
    """In-memory database of mushroom species loaded from species.csv."""

    def __init__(self):
        """Load species from the project's species.csv file."""
        import csv
        self.species: Dict[str, Dict[str, Any]] = {}
        csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "species.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    sid = row["species_id"]
                    self.species[sid] = {
                        "swedish": row["swedish_name"],
                        "english": row["english_name"],
                        "scientific": row["scientific_name"],
                        "edible": row.get("edible", "FALSE").upper() == "TRUE",
                        "toxicity": row.get("toxicity_level", "UNKNOWN"),
                        "traits": {},
                    }
        else:
            logger.warning("species.csv not found at %s — LLM species list will be empty", csv_path)
    
    def get_species(self, species_id: str) -> Optional[Dict[str, Any]]:
        """Get species by ID."""
        return self.species.get(species_id)
    
    def get_species_by_name(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Search species by English or Swedish name."""
        name_lower = name.lower()
        for species_id, data in self.species.items():
            if (data['english'].lower() == name_lower or 
                data['swedish'].lower() == name_lower or
                data['scientific'].lower() == name_lower):
                return (species_id, data)
        return None
    
    def get_all_species(self) -> Dict[str, Dict[str, Any]]:
        """Get all species."""
        return self.species
    
    def get_species_list_formatted(self) -> str:
        """Get formatted species list for prompt."""
        lines = []
        for i, (species_id, data) in enumerate(self.species.items(), 1):
            lines.append(
                f"{i}. {data['english']} ({data['swedish']}) - {data['scientific']} "
                f"[{'EDIBLE' if data['edible'] else 'TOXIC: ' + data['toxicity']}]"
            )
        return '\n'.join(lines)


class UnifiedPredictionResult:
    """Result from the unified LLM classifier."""

    def __init__(
        self,
        top_species: str,
        top_confidence: float,
        predictions: List[Tuple[str, float, str]],
        reasoning: str,
        safety_warnings: List[str],
        model_used: str,
        processing_time_ms: float,
        needs_clarification: bool = False,
        clarification_question: Optional[str] = None,
        agreement_state: str = "inconclusive",
        all_signals: Optional[Dict[str, Any]] = None,
    ):
        self.top_species = top_species
        self.top_confidence = top_confidence
        self.predictions = predictions
        self.reasoning = reasoning
        self.safety_warnings = safety_warnings
        self.model_used = model_used
        self.processing_time_ms = processing_time_ms
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question
        self.agreement_state = agreement_state
        self.all_signals = all_signals or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_species": self.top_species,
            "confidence": self.top_confidence,
            "predictions": [
                {"species": p[0], "confidence": p[1], "reason": p[2]}
                for p in self.predictions
            ],
            "reasoning": self.reasoning,
            "safety_warnings": self.safety_warnings,
            "model_used": self.model_used,
            "processing_time_ms": self.processing_time_ms,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "agreement_state": self.agreement_state,
            "all_signals": self.all_signals,
        }


class LLMPromptTemplate:
    """Manages system prompts and few-shot examples for mushroom classification."""

    MUSHROOM_SYSTEM_PROMPT = """You are an expert mycologist specializing in mushroom identification from the Nordic region (Sweden).
You will analyze descriptions of mushrooms and predict the most likely species based on morphological characteristics.

SAFETY DISCLAIMER: This system is for educational purposes only. Never use it as the sole basis for determining if a mushroom is safe to eat.
When in doubt, consult a professional mycologist or poison control.

Available Species ({species_count} total):
{species_list}

IDENTIFICATION GUIDELINES:
1. Consider all observable characteristics: cap shape/color, gill structure, stem, flesh, habitat, season
2. Match against the available species list only - do not suggest species outside this list
3. Provide confidence scores (0-1 scale) based on how well the description matches known traits
4. Flag any toxic or dangerous species
5. Explain your reasoning with specific morphological evidence
6. Indicate if the description is ambiguous or matches multiple species
7. Always include safety warnings for toxic species

RESPONSE FORMAT:
Provide your analysis as JSON with the following structure:
{{
    "top_prediction": {{"species": "English name", "confidence": 0.85, "reasoning": "..."}},
    "predictions": [
        {{"species": "Species 1", "confidence": 0.85, "reasoning": "Key features observed"}},
        {{"species": "Species 2", "confidence": 0.10, "reasoning": "..."}}
    ],
    "reasoning": "Overall analysis of the observation",
    "safety_warnings": ["WARNING: Species X is TOXIC if found"],
    "confidence_in_id": 0.85,
    "ambiguous": false,
    "needs_clarification": []
}}

Be precise, logical, and prioritize safety."""

    UNIFIED_SYSTEM_PROMPT = """You are an expert mycologist examining a mushroom specimen.
Your goal is to identify the species by combining your own visual analysis with evidence from available tools.

=== AVAILABLE TOOLS ===

1. vision_analysis — Your own expert visual examination of the provided images.
   This is your primary source of evidence. Trust your eyes first.

2. cnn_classifier — An independent AI vision system trained on mushroom images.
   This is a separate AI that simply produces an answer. It can be overconfident,
   especially on unusual lighting, odd angles, or out-of-distribution species.
   Treat its output as one signal among many, not as ground truth.

3. trait_extractor — A deterministic tool that measures morphological traits
   (cap color, gill attachment, stem features, etc.) from segmented images.
   Its output depends on segmentation quality and lighting. Verify against images.

4. dichotomous_key — A deterministic tool that navigates a Swedish dichotomous
   identification key using extracted traits. Requires precise trait matching;
   wrong traits lead to wrong paths. May get stuck if traits are ambiguous.

5. trait_database — A deterministic tool that compares extracted traits against
   known species profiles. Uses coarse descriptions; may miss fine distinctions.

=== YOUR REASONING PROCESS ===

1. EXAMINE the images carefully. Form your own preliminary diagnosis.
2. REVIEW the cnn_classifier output. Evaluate: is it plausible? What are its weaknesses?
3. REVIEW the extracted traits from trait_extractor. Do they match what you see?
4. REVIEW the dichotomous_key result. Does the path make sense given the traits?
5. REVIEW the trait_database comparison. Does the best match align with your visual diagnosis?
6. SYNTHESIZE: Which hypothesis has the strongest evidence across ALL sources?
   - If tools agree with your visual analysis, this strengthens confidence.
   - If tools contradict your visual analysis, critically evaluate BOTH sides.
   - Consider: what if YOU are wrong? What if a tool is wrong? Which has stronger evidence?
   - Use the dichotomous key to test competing hypotheses.
7. CONCLUDE with a final species identification. Do not simply follow the majority.
   Do not stubbornly stick to your first impression. Choose the strongest hypothesis.

{key_tree_text}

Available Species ({species_count} total):
{species_list}

RESPONSE FORMAT (strict JSON):
{{
    "top_prediction": {{
        "species": "English name",
        "confidence": 0.82,
        "reasoning": "Why this species fits best"
    }},
    "predictions": [
        {{"species": "English name", "confidence": 0.82, "reasoning": "..."}},
        {{"species": "Alternative", "confidence": 0.10, "reasoning": "..."}}
    ],
    "all_signals": {{
        "cnn": "Species name or 'uncertain'",
        "trait_extractor": "Summary of key extracted traits",
        "dichotomous_key": "Species name or 'incomplete'",
        "trait_database": "Species name or 'no_match'",
        "llm_own_diagnosis": "What you observed from the images yourself"
    }},
    "tool_evaluation": {{
        "cnn_trust": "high|medium|low",
        "cnn_why": "Brief justification",
        "traits_trust": "high|medium|low",
        "traits_why": "Brief justification",
        "key_trust": "high|medium|low",
        "key_why": "Brief justification",
        "database_trust": "high|medium|low",
        "database_why": "Brief justification"
    }},
    "agreement_state": "agree|disagree|partial|inconclusive",
    "reasoning": "Detailed analysis: visual observations, tool outputs, critical evaluation, final conclusion",
    "safety_warnings": ["Any toxicity warnings"],
    "needs_clarification": false,
    "clarification_question": null,
    "confidence_in_id": 0.82,
    "ambiguous": false
}}

Be concise but thorough. Prioritize safety."""

    FEW_SHOT_EXAMPLES = [
        {
            'observation': 'Yellow mushroom with a funnel-shaped cap. Gills are pale and decurrent. Firm, yellow flesh. Found on forest floor in mixed woods during autumn.',
            'expected_output': {
                'top': 'Chanterelle (Kantarell)',
                'confidence': 0.92,
                'key_features': ['Funnel-shaped cap', 'Yellow-orange color', 'Decurrent ridges', 'Pale gills', 'Mixed forest habitat']
            }
        },
        {
            'observation': 'Small red cap with white spots, white gills, white stem with a ring and bulbous base. Growing under birch trees in autumn.',
            'expected_output': {
                'top': 'Fly Agaric (Flugsvamp)',
                'confidence': 0.95,
                'key_features': ['Red cap with white spots', 'Free white gills', 'Stem ring and volva', 'Birch habitat'],
                'warning': 'TOXIC - Contains psychoactive compounds'
            }
        },
        {
            'observation': 'Brown convex cap with small yellow pores (not true gills). White stem with network pattern and bulbous base. Firm white flesh.',
            'expected_output': {
                'top': 'Porcini (Karljohan)',
                'confidence': 0.88,
                'key_features': ['Convex brown cap', 'Yellow pores', 'Pale network on stem', 'Bulbous base', 'White firm flesh']
            }
        }
    ]

    def __init__(self, species_db: SpeciesDatabase, key_tree_text: Optional[str] = None):
        """Initialize with species database and optional key.xml text."""
        self.species_db = species_db
        self.key_tree_text = key_tree_text or ""

    def get_system_prompt(self) -> str:
        """Get system prompt with species list."""
        species_list = self.species_db.get_species_list_formatted()
        return self.MUSHROOM_SYSTEM_PROMPT.format(
            species_list=species_list,
            species_count=len(self.species_db.get_all_species()),
        )

    def get_unified_system_prompt(self) -> str:
        """Get the unified system prompt with key tree injected."""
        species_list = self.species_db.get_species_list_formatted()
        return self.UNIFIED_SYSTEM_PROMPT.format(
            key_tree_text=self.key_tree_text,
            species_list=species_list,
            species_count=len(self.species_db.get_all_species()),
        )

    TREE_NAVIGATION_PROMPT = """You are an expert mycologist navigating a Swedish dichotomous key to identify a mushroom.

Use the observed traits and images to choose the correct answer at each question in the key.
Work through the key step-by-step until you reach a species decision.

{key_tree_text}

INSTRUCTIONS:
1. Read each question carefully.
2. Compare the observed traits against both possible answers.
3. Select the answer that best matches the mushroom.
4. Continue to the next question until you reach a final species decision.
5. If you cannot confidently answer a question, indicate uncertainty.

RESPONSE FORMAT (strict JSON):
{{
    "tree_path": [
        {{"question": "Exact question text from the key", "answer": "The answer you chose"}},
        ...
    ],
    "conclusion": "Swedish species name from the decision node",
    "confidence": 0.85,
    "reasoning": "Explain why this path matches the observed traits"
}}

Be precise. Use the exact question and answer texts from the key above."""

    def get_tree_navigation_prompt(self) -> str:
        """Get the system prompt for LLM tree navigation."""
        return self.TREE_NAVIGATION_PROMPT.format(key_tree_text=self.key_tree_text)

    def get_few_shot_examples(self) -> str:
        """Get few-shot examples for in-context learning."""
        lines = ["Examples of good observations and expected responses:\n"]
        for i, example in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Observation: {example['observation']}")
            lines.append(f"Response: {example['expected_output']}")
            lines.append("")
        return '\n'.join(lines)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    @abstractmethod
    def query(self, system_prompt: str, user_observation: str, images: Optional[List[str]] = None) -> str:
        """Query the LLM with observation and optional base64-encoded images."""
        pass


class MockLLMBackend(LLMBackend):
    """Mock LLM for testing when Ollama is not available."""

    def query(self, system_prompt: str, user_observation: str, images: Optional[List[str]] = None) -> str:
        """Return mock response based on keywords in observation."""
        observation_lower = user_observation.lower()

        if 'yellow' in observation_lower and 'funnel' in observation_lower:
            return json.dumps({
                'top_prediction': {'species': 'Chanterelle', 'confidence': 0.88},
                'predictions': [
                    {'species': 'Chanterelle', 'confidence': 0.88},
                    {'species': "Pig's Ear", 'confidence': 0.08},
                    {'species': 'Black Trumpet', 'confidence': 0.04}
                ],
                'reasoning': 'Yellow color and funnel shape are characteristic of Chanterelle',
                'safety_warnings': [],
                'confidence_in_id': 0.88,
                'ambiguous': False
            })
        elif 'red' in observation_lower and 'spots' in observation_lower:
            return json.dumps({
                'top_prediction': {'species': 'Fly Agaric', 'confidence': 0.95},
                'predictions': [
                    {'species': 'Fly Agaric', 'confidence': 0.95},
                    {'species': 'Other Amanita', 'confidence': 0.05}
                ],
                'reasoning': 'Red cap with white spots is diagnostic of Fly Agaric',
                'safety_warnings': ['TOXIC: This species contains psychoactive compounds'],
                'confidence_in_id': 0.95,
                'ambiguous': False
            })
        else:
            return json.dumps({
                'top_prediction': {'species': 'Unknown', 'confidence': 0.3},
                'predictions': [],
                'reasoning': 'Insufficient information for reliable identification',
                'safety_warnings': ['Please provide more detailed observations'],
                'confidence_in_id': 0.3,
                'ambiguous': True
            })


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend — no API key required."""

    DEFAULT_MODEL = "gemma3:12b"
    BASE_URL      = "http://localhost:11434"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        import requests as _requests
        self._requests = _requests
        self.model    = model    or os.environ.get("OLLAMA_MODEL",    self.DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", self.BASE_URL)

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            import requests as _r
            resp = _r.get(f"{cls.BASE_URL}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def query(self, system_prompt: str, user_observation: str, images: Optional[List[str]] = None) -> str:
        """Send observation to Ollama and return the model's raw response string.

        Args:
            system_prompt: System instructions.
            user_observation: User text prompt.
            images: Optional list of base64-encoded image strings.
        """
        user_msg: Dict[str, Any] = {"role": "user", "content": user_observation}
        if images:
            user_msg["images"] = images
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_msg,
            ],
            "stream": False,
            # NOTE: format="json" is disabled because small models frequently
            # returns malformed JSON for complex prompts. The _parse_response()
            # method handles both JSON and free-text fallback.
            # "format": "json",
            "options": {
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0")),
                "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "512")),
            },
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
        }
        try:
            resp = self._requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=float(os.environ.get("OLLAMA_TIMEOUT", "300")),
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            raise


class LLMClassifier:
    """Main LLM classifier integrating prompts, backends, and response parsing."""

    def __init__(self, backend_type: str = 'ollama',
                 ollama_model: Optional[str] = None,
                 key_tree_text: Optional[str] = None):
        """
        Initialize classifier with specified backend.

        Args:
            backend_type: 'ollama' or 'mock'
            ollama_model: Ollama model name (default: gemma3:12b)
            key_tree_text: Optional key.xml text to inject into unified prompts
        """
        self.species_db = SpeciesDatabase()
        self.prompt_template = LLMPromptTemplate(self.species_db, key_tree_text=key_tree_text)
        self.backend_type = backend_type

        if backend_type == 'mock':
            self.backend = MockLLMBackend()
        elif backend_type == 'ollama':
            self.backend = OllamaBackend(model=ollama_model)
        else:
            raise ValueError(f'Unknown backend type: {backend_type}')

        logger.info(f'LLMClassifier initialized with {backend_type} backend')

    def classify(self, observation: str, context: Optional[Dict[str, str]] = None) -> PredictionResult:
        """
        Classify mushroom from natural language observation.

        Args:
            observation: Natural language description of mushroom
            context: Optional context (habitat, season, substrate)

        Returns:
            PredictionResult with standardized format
        """
        import time
        start_time = time.time()

        system_prompt = self.prompt_template.get_system_prompt()
        user_input = self._format_user_input(observation, context)

        try:
            response = self.backend.query(system_prompt, user_input)
            result = self._parse_response(response)

            processing_time = (time.time() - start_time) * 1000

            return PredictionResult(
                top_species=result.get('top_prediction', {}).get('species', 'Unknown'),
                top_confidence=float(result.get('confidence_in_id', 0.0)),
                predictions=self._format_predictions(result.get('predictions', [])),
                reasoning=result.get('reasoning', 'No reasoning provided'),
                safety_warnings=result.get('safety_warnings', []),
                model_used=self.backend_type,
                processing_time_ms=processing_time
            )
        except Exception as e:
            logger.error(f'Classification error: {e}')
            return PredictionResult(
                top_species='Error',
                top_confidence=0.0,
                predictions=[],
                reasoning=f'Error during classification: {str(e)}',
                safety_warnings=['Classification failed - consult expert'],
                model_used=self.backend_type,
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def navigate_tree(
        self,
        visible_traits: Dict[str, Any],
        cnn_prediction: Dict[str, Any],
        case_info: Optional[Dict[str, Any]] = None,
        images_b64: Optional[List[str]] = None,
        pre_answers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """One-shot LLM tree navigation.

        The LLM receives the tree structure + observed traits + CNN hint
        and returns its chosen path through the dichotomous key.

        Args:
            visible_traits: Extracted visual traits.
            cnn_prediction: CNN top prediction (optional hint).
            case_info: Morphological case info.
            images_b64: Optional base64-encoded images.
            pre_answers: Optional oracle pre_answers to inject as the correct path.

        Returns:
            Dict with keys: tree_path (list), conclusion (str), confidence (float),
            reasoning (str), raw_response (str).
        """
        import time
        start_time = time.time()

        system_prompt = self.prompt_template.get_tree_navigation_prompt()

        # Build user input with traits and optional CNN hint
        lines = ["=== OBSERVED TRAITS ==="]
        for k, v in visible_traits.items():
            lines.append(f"  {k}: {v}")

        if cnn_prediction and cnn_prediction.get("species"):
            lines.append(f"\nCNN HINT (uncertain, use with caution):")
            lines.append(f"  Predicted: {cnn_prediction['species']} (confidence: {cnn_prediction.get('confidence', 0)})")

        if case_info:
            lines.append(f"\nMORPHOLOGICAL CASE: {case_info.get('case', 'unknown')}")
            lines.append(f"  Detected parts: {case_info.get('detected_parts', [])}")

        if pre_answers:
            lines.append("\n=== ORACLE PATH (use these answers) ===")
            for q, a in pre_answers.items():
                lines.append(f"  Q: {q} → A: {a}")
            lines.append("Navigate the tree using the oracle answers above and report the final species.")
        else:
            lines.append("\nNavigate the key using the observed traits. Reach a final species decision.")

        user_input = "\n".join(lines)

        try:
            response = self.backend.query(system_prompt, user_input, images=images_b64)
            parsed = self._parse_response(response)

            processing_time = (time.time() - start_time) * 1000

            tree_path = parsed.get("tree_path", [])
            if isinstance(tree_path, str):
                # Sometimes LLM returns a string instead of list
                tree_path = []

            return {
                "tree_path": tree_path,
                "conclusion": parsed.get("conclusion", "Unknown"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
                "raw_response": response,
                "processing_time_ms": processing_time,
            }
        except Exception as exc:
            logger.error(f"Tree navigation error: {exc}")
            return {
                "tree_path": [],
                "conclusion": "Error",
                "confidence": 0.0,
                "reasoning": f"Error: {exc}",
                "raw_response": "",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    def unified_classify(
        self,
        visible_traits: Dict[str, Any],
        cnn_prediction: Dict[str, Any],
        tree_result: Dict[str, Any],
        db_result: Dict[str, Any],
        case_info: Optional[Dict[str, Any]] = None,
        image_descriptions: Optional[List[str]] = None,
        images_b64: Optional[List[str]] = None,
    ) -> UnifiedPredictionResult:
        """
        Unified classification using ALL available signals.

        Args:
            visible_traits: Extracted visual traits
            cnn_prediction: Output from CNN.predict_with_uncertainty()
            tree_result: Output from KeyTreeEngine traversal
            db_result: Output from TraitDatabaseComparator.compare()
            case_info: Output from detect_case()
            image_descriptions: Optional text descriptions of each image

        Returns:
            UnifiedPredictionResult with all signals and agreement evaluation
        """
        import time
        start_time = time.time()

        system_prompt = self.prompt_template.get_unified_system_prompt()
        user_input = self._build_unified_user_input(
            visible_traits, cnn_prediction, tree_result, db_result,
            case_info, image_descriptions,
        )

        try:
            response = self.backend.query(system_prompt, user_input, images=images_b64)
            result = self._parse_response(response)

            processing_time = (time.time() - start_time) * 1000

            # Extract all_signals if present
            all_signals = result.get("all_signals", {})
            agreement = result.get("agreement_state", "inconclusive")
            needs_clarification = result.get("needs_clarification", False)
            clarification = result.get("clarification_question")

            # Gracefully handle multiple possible JSON key layouts
            top_pred = result.get('top_prediction', {}) or {}
            top_species = (
                top_pred.get('species')
                or result.get('species')
                or result.get('top_species')
                or 'Unknown'
            )
            top_confidence = float(
                top_pred.get('confidence')
                or result.get('confidence')
                or result.get('confidence_in_id')
                or result.get('top_confidence')
                or 0.0
            )

            # Build predictions list if missing
            raw_predictions = result.get('predictions', [])
            if not raw_predictions and top_species not in ('Unknown', 'Error', 'Unable to parse'):
                raw_predictions = [{
                    'species': top_species,
                    'confidence': top_confidence,
                    'reasoning': result.get('reasoning', '')
                }]

            return UnifiedPredictionResult(
                top_species=top_species,
                top_confidence=top_confidence,
                predictions=self._format_predictions(raw_predictions),
                reasoning=result.get('reasoning', 'No reasoning provided'),
                safety_warnings=result.get('safety_warnings', []),
                model_used=self.backend_type,
                processing_time_ms=processing_time,
                needs_clarification=needs_clarification,
                clarification_question=clarification,
                agreement_state=agreement,
                all_signals=all_signals,
            )
        except Exception as e:
            logger.error(f'Unified classification error: {e}')
            processing_time = (time.time() - start_time) * 1000
            return UnifiedPredictionResult(
                top_species='Error',
                top_confidence=0.0,
                predictions=[],
                reasoning=f'Error during unified classification: {str(e)}',
                safety_warnings=['Classification failed - consult expert'],
                model_used=self.backend_type,
                processing_time_ms=processing_time,
                needs_clarification=False,
                agreement_state="inconclusive",
            )

    def _build_unified_user_input(
        self,
        visible_traits: Dict[str, Any],
        cnn_prediction: Dict[str, Any],
        tree_result: Dict[str, Any],
        db_result: Dict[str, Any],
        case_info: Optional[Dict[str, Any]],
        image_descriptions: Optional[List[str]],
    ) -> str:
        """Build user prompt: LLM as central reasoner, subsystems as tools."""
        lines: List[str] = []
        lines.append("=== SPECIMEN IMAGES ===")
        lines.append("(Examine both images carefully before reviewing the tool outputs below.)\n")

        if case_info:
            lines.append(f"MORPHOLOGICAL CASE: {case_info.get('case', 'unknown')}")
            lines.append(f"Detected parts: {case_info.get('detected_parts', [])}\n")

        if image_descriptions:
            for i, desc in enumerate(image_descriptions, 1):
                lines.append(f"IMAGE {i} DESCRIPTION: {desc}")
            lines.append("")

        lines.append("=== YOUR PRELIMINARY DIAGNOSIS ===")
        lines.append("(Look at the images first. What species does your own visual analysis suggest?\n")
        lines.append("Form your own opinion before reading the tool outputs below.)\n")

        # Tool 1: CNN (independent AI)
        lines.append("=== TOOL OUTPUT: cnn_classifier ===")
        lines.append(f"  Prediction: {cnn_prediction.get('species', 'None')} (confidence: {cnn_prediction.get('confidence', 0.0):.4f})")
        lines.append(f"  Conclusive: {cnn_prediction.get('conclusive', False)}")
        if cnn_prediction.get('uncertainty_reason'):
            lines.append(f"  Note: {cnn_prediction['uncertainty_reason']}")
        top5 = cnn_prediction.get('top_5', [])
        if top5:
            lines.append(f"  Top-5 alternatives: {top5}")
        lines.append("  WARNING: CNNs can be overconfident, especially on unusual specimens or poor lighting.\n")

        # Tool 2: dichotomous_key
        lines.append("=== TOOL OUTPUT: dichotomous_key ===")
        status = tree_result.get('status', 'unknown')
        lines.append(f"  Status: {status}")
        if status == 'conclusion':
            lines.append(f"  Conclusion: {tree_result.get('species', 'unknown')}")
            path = tree_result.get('path', [])
            if path:
                lines.append(f"  Path taken: {' → '.join(path)}")
        elif status == 'question':
            lines.append(f"  Stuck at question: {tree_result.get('question', '')}")
            lines.append(f"  Options: {tree_result.get('options', [])}")
            lines.append("  WARNING: Key traversal is incomplete — trait extraction may be missing critical features.")
        lines.append("  WARNING: The key requires precise trait matching; wrong traits lead to wrong paths.\n")

        # Tool 3: trait_database
        lines.append("=== TOOL OUTPUT: trait_database ===")
        db_status = db_result.get('status', 'unknown')
        lines.append(f"  Status: {db_status}")
        if db_status == 'ok':
            cand = db_result.get('candidate', {})
            lines.append(f"  Best match: {cand.get('swedish_name', 'unknown')} ({cand.get('english_name', 'unknown')})")
            tm = db_result.get('trait_match', {})
            lines.append(f"  Trait match score: {tm.get('score', 0.0)}")
            lines.append(f"  Conflicts: {len(tm.get('conflicts', []))}")
            lookalikes = db_result.get('lookalikes', [])
            if lookalikes:
                lines.append(f"  Lookalikes found: {len(lookalikes)}")
                for la in lookalikes[:3]:
                    lines.append(f"    - {la.get('swedish_name', '')} (toxicity: {la.get('toxicity_level', 'unknown')})")
        else:
            lines.append("  No database match found.")
        lines.append("  WARNING: Database matching uses coarse trait descriptions and may miss fine distinctions.\n")

        # Traits reference
        lines.append("=== EXTRACTED TRAITS (for reference) ===")
        for k, v in visible_traits.items():
            if k == "colour_ratios":
                lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

        lines.append("=== YOUR FINAL IDENTIFICATION ===")
        lines.append("Now make YOUR final call.")
        lines.append("If the tools agree with what you see, use that to strengthen your confidence.")
        lines.append("If they contradict your visual analysis, critically evaluate BOTH your prediction and the tool predictions.")
        lines.append("Consider: what if YOU are wrong? What if a tool is wrong? Which prediction has the stronger evidence?")
        lines.append("Use the decision key to test which hypothesis fits best. Choose the strongest hypothesis, even if it changes your mind.")
        lines.append("Do not simply follow the majority of tools, and do not stubbornly stick to your first impression.")
        lines.append("Return your answer in the JSON format specified in your instructions.\n")

        return "\n".join(lines)

    def _format_user_input(self, observation: str, context: Optional[Dict[str, str]] = None) -> str:
        """Format user observation with optional context."""
        lines = [f"Mushroom Observation: {observation}"]

        if context:
            if context.get('habitat'):
                lines.append(f"Habitat: {context['habitat']}")
            if context.get('season'):
                lines.append(f"Season: {context['season']}")
            if context.get('substrate'):
                lines.append(f"Substrate: {context['substrate']}")

        lines.append("\nBased on this description, identify the most likely species from the available list.")
        return '\n'.join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response (handles JSON, markdown blocks, free text, truncation)."""
        raw = response.strip()

        # 1. Strip markdown code fences
        cleaned = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # 2. Try direct JSON parse
        for candidate in (raw, cleaned):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # 3. Find all JSON objects and try the largest ones first
        matches = re.findall(r"\{[\s\S]*?\}", cleaned)
        if matches:
            # Sort by length descending — prefer largest (most complete) object
            for candidate in sorted(matches, key=len, reverse=True):
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, ValueError):
                    continue

        # 4. Fallback: regex extract species name and confidence from free text
        logger.warning(
            'JSON parse failed (%d chars). Falling back to regex extraction.',
            len(cleaned)
        )

        species_match = re.search(
            r'(?:species|name|top_prediction)["\'\s]*[:=]\s*["\']?([^"\'\n,;{}]+)',
            cleaned,
            re.IGNORECASE,
        )
        species = species_match.group(1).strip() if species_match else "Unknown"

        conf_match = re.search(
            r'(?:confidence|conf|confidence_in_id)["\'\s]*[:=]\s*([0-9]*\.?[0-9]+)',
            cleaned,
            re.IGNORECASE,
        )
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        return {
            'top_prediction': {'species': species, 'confidence': confidence},
            'predictions': [{'species': species, 'confidence': confidence}],
            'reasoning': cleaned,
            'safety_warnings': [],
            'confidence_in_id': confidence,
            'ambiguous': species == "Unknown",
        }

    def _format_predictions(self, predictions: List[Dict[str, Any]]) -> List[Tuple[str, float, str]]:
        """Format predictions to standard tuple format."""
        return [
            (p.get('species', 'Unknown'), float(p.get('confidence', 0.0)), p.get('reasoning', ''))
            for p in predictions[:5]  # Top 5
        ]
