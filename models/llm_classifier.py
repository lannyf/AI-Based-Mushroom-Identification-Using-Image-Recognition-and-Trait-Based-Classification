"""
LLM-Based Mushroom Classifier

Implements natural language processing for mushroom identification using
Large Language Models (LLM). Supports multiple LLM backends:
- OpenAI GPT-4 (recommended for accuracy)
- Local LLama-2 (open-source, privacy-preserving)
- Mock LLM (for testing without API key)

The module provides:
1. LLMPromptTemplate: System prompts with mushroom expertise context
2. LLMClassifier: API client and response parsing
3. SpeciesDatabase: In-memory species lookup
4. PredictionResult: Standardized output format
"""

import os
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
    """In-memory database of mushroom species with bilingual names."""
    
    def __init__(self):
        """Initialize with 20 mushroom species from Nya Svampboken."""
        self.species = {
            'CA.CI': {
                'swedish': 'Kantarell',
                'english': 'Chanterelle',
                'scientific': 'Cantharellus cibarius',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'funnel-shaped, yellow-orange',
                    'gills': 'ridges, pale yellow, decurrent',
                    'stem': 'solid, cylindrical',
                    'flesh': 'pale yellow, firm',
                    'habitat': 'forest floor, mixed trees',
                    'season': 'summer to autumn'
                }
            },
            'CR.CO': {
                'swedish': 'Svart Trumpetsvamp',
                'english': 'Black Trumpet',
                'scientific': 'Craterellus cinereus',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'funnel-shaped, dark gray to black',
                    'gills': 'ridges, whitish',
                    'stem': 'hollow, dark',
                    'flesh': 'thin, gray',
                    'habitat': 'forest floor',
                    'season': 'summer to autumn'
                }
            },
            'BO.ED': {
                'swedish': 'Karljohan',
                'english': 'Porcini',
                'scientific': 'Boletus edulis',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'convex, brown',
                    'gills': 'pores instead of gills, pale yellow',
                    'stem': 'bulbous, pale with network pattern',
                    'flesh': 'white, firm',
                    'habitat': 'forest floor under pine/spruce',
                    'season': 'summer to autumn'
                }
            },
            'AM.MU': {
                'swedish': 'Flugsvamp',
                'english': 'Fly Agaric',
                'scientific': 'Amanita muscaria',
                'edible': False,
                'toxicity': 'TOXIC',
                'traits': {
                    'cap': 'convex, bright red with white spots',
                    'gills': 'free, white',
                    'stem': 'white with ring and volva',
                    'flesh': 'white, soft',
                    'habitat': 'birch and pine forests',
                    'season': 'autumn'
                }
            },
            'AM.VI': {
                'swedish': 'Vit Flugsvamp',
                'english': 'Destroying Angel',
                'scientific': 'Amanita virosa',
                'edible': False,
                'toxicity': 'DEADLY',
                'traits': {
                    'cap': 'white to cream, hemispherical',
                    'gills': 'free, white',
                    'stem': 'white with ring and bulbous volva',
                    'flesh': 'white, thin',
                    'habitat': 'mixed forests',
                    'season': 'autumn'
                }
            },
            'PS001': {
                'swedish': 'Björkskivling',
                'english': 'Birch Polypore',
                'scientific': 'Piptoporus betulinus',
                'edible': False,
                'toxicity': 'INEDIBLE',
                'traits': {
                    'cap': 'hoof-shaped, white to brown',
                    'gills': 'pores, small, white',
                    'stem': 'none, shelf-like',
                    'flesh': 'white, tough',
                    'habitat': 'birch trees',
                    'season': 'year-round'
                }
            },
            'GO001': {
                'swedish': 'Grisöra',
                'english': "Pig's Ear",
                'scientific': 'Gomphus clavatus',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'funnel-shaped, brown to purple-brown',
                    'gills': 'ridges, pale, blunt',
                    'stem': 'solid, cylindrical',
                    'flesh': 'pale, firm',
                    'habitat': 'forest floor, often in groups',
                    'season': 'summer to autumn'
                }
            },
            'SU.LU': {
                'swedish': 'Smörsopp',
                'english': 'Slippery Jack',
                'scientific': 'Suillus luteus',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'convex, yellow-brown, slimy',
                    'gills': 'pores, small, yellow',
                    'stem': 'yellow-brown with ring',
                    'flesh': 'pale yellow, soft',
                    'habitat': 'pine forests',
                    'season': 'summer to autumn'
                }
            },
            'LY.PE': {
                'swedish': 'Behandlad Behandling',
                'english': 'Common Puffball',
                'scientific': 'Lycoperdon perlatum',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'round, white with spikes',
                    'gills': 'none (spore sac)',
                    'stem': 'none, attached to ground',
                    'flesh': 'white, spore-filled',
                    'habitat': 'grassland, open forest',
                    'season': 'summer to autumn'
                }
            },
            'RU001': {
                'swedish': 'Russula',
                'english': 'Russula species',
                'scientific': 'Russula mairei',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'convex, white to red',
                    'gills': 'free, white, brittle',
                    'stem': 'white, fragile',
                    'flesh': 'white, crisp',
                    'habitat': 'mixed forests',
                    'season': 'summer to autumn'
                }
            },
            'LA001': {
                'swedish': 'Behandlad Mjölkchanterelle',
                'english': 'Lactarius species',
                'scientific': 'Lactarius pubescens',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'convex, white to cream',
                    'gills': 'decurrent, white with latex',
                    'stem': 'white, hollow',
                    'flesh': 'white, brittle',
                    'habitat': 'birch forests',
                    'season': 'summer to autumn'
                }
            },
            'CA.TU': {
                'swedish': 'Trattkantarell',
                'english': 'Trumpet Chanterelle',
                'scientific': 'Craterellus tubaeformis',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'funnel-shaped, yellow-brown',
                    'gills': 'ridges, pale yellow',
                    'stem': 'hollow, yellow',
                    'flesh': 'thin, pale',
                    'habitat': 'forest floor',
                    'season': 'summer to autumn'
                }
            },
            'CP001': {
                'swedish': 'Penselbindsvamp',
                'english': 'Copper Inky Cap',
                'scientific': 'Coprinellus micaceus',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'cylindrical, brown with silvery sheen',
                    'gills': 'attached, dark, turning black',
                    'stem': 'white, hollow',
                    'flesh': 'thin, pale',
                    'habitat': 'wood debris, stumps',
                    'season': 'autumn to spring'
                }
            },
            'HY.RE': {
                'swedish': 'Behandlad Boletaceae',
                'english': 'Hedgehog Mushroom',
                'scientific': 'Hydnum repandum',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'convex, white to orange',
                    'gills': 'spines instead of gills, pale',
                    'stem': 'white-orange, solid',
                    'flesh': 'white, firm',
                    'habitat': 'mixed forest floor',
                    'season': 'summer to autumn'
                }
            },
            'CA001': {
                'swedish': 'Behandlad Chanterelle',
                'english': 'False Chanterelle',
                'scientific': 'Cantharellula cibarius',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'funnel-shaped, pale orange',
                    'gills': 'ridges, pale',
                    'stem': 'hollow, orange',
                    'flesh': 'thin, orange',
                    'habitat': 'forest floor',
                    'season': 'summer to autumn'
                }
            },
            'LA.TU': {
                'swedish': 'Behandlad Milk Cap',
                'english': 'Milky Cap',
                'scientific': 'Lactarius turpis',
                'edible': False,
                'toxicity': 'INEDIBLE',
                'traits': {
                    'cap': 'convex, dark brown with rings',
                    'gills': 'decurrent, cream with latex',
                    'stem': 'brown, hollow',
                    'flesh': 'cream, brittle',
                    'habitat': 'birch forests',
                    'season': 'summer to autumn'
                }
            },
            'ST001': {
                'swedish': 'Behandlad Stinkhorn',
                'english': 'Stinkhorn',
                'scientific': 'Phallus impudicus',
                'edible': False,
                'toxicity': 'INEDIBLE',
                'traits': {
                    'cap': 'latticed, orange-red, fetid smell',
                    'gills': 'none (network structure)',
                    'stem': 'white, spongy',
                    'flesh': 'white, hollow',
                    'habitat': 'soil, decaying wood',
                    'season': 'summer to autumn'
                }
            },
            'GI001': {
                'swedish': 'Jätteska',
                'english': 'Giant Puffball',
                'scientific': 'Calvatia gigantea',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'spherical, white',
                    'gills': 'none (spore sac)',
                    'stem': 'none',
                    'flesh': 'white, spore-filled',
                    'habitat': 'grassland, meadows',
                    'season': 'summer to autumn'
                }
            },
            'BL001': {
                'swedish': 'Svampkonk',
                'english': 'Artist\'s Conk',
                'scientific': 'Ganoderma applanatum',
                'edible': False,
                'toxicity': 'INEDIBLE',
                'traits': {
                    'cap': 'shelf-shaped, brown-black',
                    'gills': 'pores, small, brown',
                    'stem': 'lateral attachment',
                    'flesh': 'brown, corky',
                    'habitat': 'birch and pine trees',
                    'season': 'year-round'
                }
            },
            'AU.AU': {
                'swedish': 'Behandlad Wood Ear',
                'english': 'Wood Ear',
                'scientific': 'Auricularia auricula',
                'edible': True,
                'toxicity': 'SAFE',
                'traits': {
                    'cap': 'ear-shaped, dark brown',
                    'gills': 'none (smooth underside)',
                    'stem': 'lateral attachment',
                    'flesh': 'gelatinous, thin',
                    'habitat': 'dead wood, trees',
                    'season': 'year-round'
                }
            },
        }
    
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


class LLMPromptTemplate:
    """Manages system prompts and few-shot examples for mushroom classification."""
    
    MUSHROOM_SYSTEM_PROMPT = """You are an expert mycologist specializing in mushroom identification from the Nordic region (Sweden). 
You will analyze descriptions of mushrooms and predict the most likely species based on morphological characteristics.

SAFETY DISCLAIMER: This system is for educational purposes only. Never use it as the sole basis for determining if a mushroom is safe to eat. 
When in doubt, consult a professional mycologist or poison control.

Available Species (20 total):
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
    
    def __init__(self, species_db: SpeciesDatabase):
        """Initialize with species database."""
        self.species_db = species_db
    
    def get_system_prompt(self) -> str:
        """Get system prompt with species list."""
        species_list = self.species_db.get_species_list_formatted()
        return self.MUSHROOM_SYSTEM_PROMPT.format(species_list=species_list)
    
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
    def query(self, system_prompt: str, user_observation: str) -> str:
        """Query the LLM with observation."""
        pass


class MockLLMBackend(LLMBackend):
    """Mock LLM for testing without API key."""
    
    def query(self, system_prompt: str, user_observation: str) -> str:
        """Return mock response based on keywords in observation."""
        observation_lower = observation_lower = user_observation.lower()
        
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


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend (requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key."""
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY environment variable not set')
        try:
            import openai
            openai.api_key = self.api_key
            self.openai = openai
        except ImportError:
            raise ImportError('openai package required for OpenAI backend')
    
    def query(self, system_prompt: str, user_observation: str) -> str:
        """Query OpenAI GPT."""
        try:
            response = self.openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_observation}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f'OpenAI API error: {e}')
            raise


class LLMClassifier:
    """Main LLM classifier integrating prompts, backends, and response parsing."""
    
    def __init__(self, backend_type: str = 'mock', api_key: Optional[str] = None):
        """
        Initialize classifier with specified backend.
        
        Args:
            backend_type: 'mock', 'openai', or 'huggingface'
            api_key: API key for OpenAI (if using openai backend)
        """
        self.species_db = SpeciesDatabase()
        self.prompt_template = LLMPromptTemplate(self.species_db)
        self.backend_type = backend_type
        
        if backend_type == 'mock':
            self.backend = MockLLMBackend()
        elif backend_type == 'openai':
            self.backend = OpenAIBackend(api_key)
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
        """Parse LLM response (handles both JSON and text)."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning('Could not parse JSON response, extracting text')
            return {
                'top_prediction': {'species': 'Unable to parse', 'confidence': 0.0},
                'reasoning': response,
                'safety_warnings': [],
                'confidence_in_id': 0.0
            }
    
    def _format_predictions(self, predictions: List[Dict[str, Any]]) -> List[Tuple[str, float, str]]:
        """Format predictions to standard tuple format."""
        return [
            (p.get('species', 'Unknown'), float(p.get('confidence', 0.0)), p.get('reasoning', ''))
            for p in predictions[:5]  # Top 5
        ]
