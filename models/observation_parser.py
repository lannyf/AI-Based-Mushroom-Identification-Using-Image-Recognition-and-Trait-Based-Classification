"""
Observation Parser for Natural Language Mushroom Descriptions

Extracts structured traits from natural language observations and normalizes
them for use with the trait-based classifier. Handles:
- Morphological features (cap, gills, stem, flesh, habitat, season)
- Partial or ambiguous descriptions
- Contextual information (location, season, substrate)
- Confidence levels from user
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TraitCategory(Enum):
    """Mushroom trait categories."""
    CAP = "cap"
    GILLS = "gills"
    STEM = "stem"
    FLESH = "flesh"
    HABITAT = "habitat"
    SEASON = "season"
    GROWTH = "growth"


@dataclass
class ParsedTrait:
    """Single extracted trait with confidence."""
    category: TraitCategory
    value: str
    raw_text: str
    confidence: float  # 0-1, how certain the user is
    is_inferred: bool = False


@dataclass
class ParsedObservation:
    """Complete parsed observation with extracted traits."""
    raw_description: str
    traits: List[ParsedTrait]
    identified_traits: Dict[str, str]  # category -> value mapping
    missing_traits: List[str]
    ambiguous_traits: List[str]
    overall_confidence: float
    context: Dict[str, str]  # habitat, season, substrate
    quality_score: float  # 0-1, how complete/reliable the observation is


class ObservationParser:
    """Parse natural language mushroom observations into structured traits."""
    
    # Common color terms
    COLOR_PATTERNS = {
        'yellow': ['yellow', 'golden', 'golden-yellow', 'sulfur'],
        'red': ['red', 'crimson', 'scarlet', 'reddish'],
        'brown': ['brown', 'tan', 'chocolate', 'cinnamon', 'dark brown'],
        'white': ['white', 'cream', 'pale', 'ivory'],
        'orange': ['orange', 'orange-red', 'orange-brown'],
        'gray': ['gray', 'grey', 'dark gray', 'silvery'],
        'black': ['black', 'dark', 'ebony'],
        'green': ['green', 'olive', 'greenish'],
        'purple': ['purple', 'violet', 'lavender'],
    }
    
    # Cap shape terms
    CAP_SHAPE_PATTERNS = {
        'convex': ['convex', 'rounded', 'dome-shaped', 'umbrella-like'],
        'flat': ['flat', 'plane', 'shield-shaped'],
        'funnel': ['funnel', 'funnel-shaped', 'trumpet', 'vase-like', 'cup-shaped'],
        'hemispherical': ['hemispherical', 'dome', 'bell-shaped', 'umbonate'],
        'conical': ['conical', 'cone-shaped', 'pointed'],
    }
    
    # Gill-related terms
    GILL_PATTERNS = {
        'free': ['free', 'not attached', 'attached to stem'],
        'attached': ['attached', 'adnate', 'sinuate'],
        'decurrent': ['decurrent', 'running down', 'notched'],
        'spacing': {
            'crowded': ['crowded', 'close', 'densely packed'],
            'distant': ['distant', 'far apart', 'widely spaced'],
        },
        'color': COLOR_PATTERNS,
    }
    
    # Habitat terms
    HABITAT_PATTERNS = {
        'forest': ['forest', 'wood', 'woodland', 'grove', 'trees'],
        'grassland': ['grass', 'field', 'meadow', 'lawn', 'open'],
        'soil': ['soil', 'ground', 'leaf litter', 'dirt'],
        'wood': ['wood', 'log', 'stump', 'dead tree', 'decaying'],
        'mossy': ['moss', 'mossy', 'damp'],
    }
    
    # Season terms
    SEASON_PATTERNS = {
        'spring': ['spring', 'april', 'may', 'march', 'early'],
        'summer': ['summer', 'june', 'july', 'august', 'warm'],
        'autumn': ['autumn', 'fall', 'september', 'october', 'november'],
        'winter': ['winter', 'december', 'january', 'february', 'cold'],
    }
    
    # Tree association patterns
    TREE_PATTERNS = {
        'birch': ['birch', 'betula'],
        'pine': ['pine', 'pinus'],
        'spruce': ['spruce', 'picea'],
        'oak': ['oak', 'quercus'],
        'beech': ['beech', 'fagus'],
        'mixed': ['mixed', 'various', 'deciduous', 'coniferous'],
    }
    
    # Quantifiers for confidence
    CONFIDENCE_MODIFIERS = {
        'certain': ['definitely', 'clearly', 'obviously', 'certainly'],
        'likely': ['likely', 'probably', 'seems', 'appears'],
        'uncertain': ['might', 'could', 'possibly', 'maybe', 'perhaps', 'uncertain'],
        'negative': ['not', 'no', 'absent', 'none'],
    }
    
    def __init__(self):
        """Initialize parser."""
        self.required_traits = ['cap', 'gills', 'stem', 'habitat']
        self.optional_traits = ['flesh', 'season', 'growth']
    
    def parse(self, observation: str, context: Optional[Dict[str, str]] = None) -> ParsedObservation:
        """
        Parse natural language observation into structured traits.
        
        Args:
            observation: Raw text description of mushroom
            context: Optional context dict with 'habitat', 'season', 'substrate' keys
        
        Returns:
            ParsedObservation with extracted traits
        """
        traits = []
        identified_traits = {}
        ambiguous_traits = []
        
        # Extract cap characteristics
        cap_traits = self._extract_cap_traits(observation)
        traits.extend(cap_traits)
        if cap_traits:
            identified_traits['cap'] = ', '.join([t.value for t in cap_traits])
        else:
            ambiguous_traits.append('cap')
        
        # Extract gill characteristics
        gill_traits = self._extract_gill_traits(observation)
        traits.extend(gill_traits)
        if gill_traits:
            identified_traits['gills'] = ', '.join([t.value for t in gill_traits])
        else:
            ambiguous_traits.append('gills')
        
        # Extract stem characteristics
        stem_traits = self._extract_stem_traits(observation)
        traits.extend(stem_traits)
        if stem_traits:
            identified_traits['stem'] = ', '.join([t.value for t in stem_traits])
        else:
            ambiguous_traits.append('stem')
        
        # Extract flesh characteristics
        flesh_traits = self._extract_flesh_traits(observation)
        traits.extend(flesh_traits)
        if flesh_traits:
            identified_traits['flesh'] = ', '.join([t.value for t in flesh_traits])
        
        # Extract habitat information
        habitat_traits = self._extract_habitat_traits(observation, context)
        traits.extend(habitat_traits)
        if habitat_traits:
            identified_traits['habitat'] = ', '.join([t.value for t in habitat_traits])
        
        # Extract season information
        season_traits = self._extract_season_traits(observation, context)
        traits.extend(season_traits)
        if season_traits:
            identified_traits['season'] = ', '.join([t.value for t in season_traits])
        
        # Calculate missing traits
        missing_traits = [t for t in self.required_traits if t not in identified_traits]
        
        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(traits, identified_traits)
        
        # Calculate quality score (0-1)
        quality_score = self._calculate_quality(identified_traits, ambiguous_traits)
        
        # Prepare context
        parsed_context = context or {}
        
        return ParsedObservation(
            raw_description=observation,
            traits=traits,
            identified_traits=identified_traits,
            missing_traits=missing_traits,
            ambiguous_traits=ambiguous_traits,
            overall_confidence=overall_confidence,
            context=parsed_context,
            quality_score=quality_score
        )
    
    def _extract_cap_traits(self, text: str) -> List[ParsedTrait]:
        """Extract cap-related traits."""
        traits = []
        text_lower = text.lower()
        
        # Shape
        for shape, patterns in self.CAP_SHAPE_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                traits.append(ParsedTrait(
                    category=TraitCategory.CAP,
                    value=f'shape: {shape}',
                    raw_text=text,
                    confidence=0.8
                ))
                break
        
        # Color
        for color, patterns in self.COLOR_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                traits.append(ParsedTrait(
                    category=TraitCategory.CAP,
                    value=f'color: {color}',
                    raw_text=text,
                    confidence=0.7
                ))
                break
        
        return traits
    
    def _extract_gill_traits(self, text: str) -> List[ParsedTrait]:
        """Extract gill-related traits."""
        traits = []
        text_lower = text.lower()
        
        # Attachment
        if any(p in text_lower for p in self.GILL_PATTERNS['decurrent']):
            traits.append(ParsedTrait(
                category=TraitCategory.GILLS,
                value='attachment: decurrent',
                raw_text=text,
                confidence=0.8
            ))
        elif any(p in text_lower for p in self.GILL_PATTERNS['free']):
            traits.append(ParsedTrait(
                category=TraitCategory.GILLS,
                value='attachment: free',
                raw_text=text,
                confidence=0.8
            ))
        
        # Color
        for color, patterns in self.COLOR_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                # Check if it's specifically about gills
                if any(word in text_lower for word in ['gill', 'lamella', 'ridge']):
                    traits.append(ParsedTrait(
                        category=TraitCategory.GILLS,
                        value=f'color: {color}',
                        raw_text=text,
                        confidence=0.7
                    ))
                    break
        
        return traits
    
    def _extract_stem_traits(self, text: str) -> List[ParsedTrait]:
        """Extract stem-related traits."""
        traits = []
        text_lower = text.lower()
        
        # Check for stem mentions
        if 'stem' not in text_lower and 'stalk' not in text_lower:
            return traits
        
        # Attachment style
        if 'bulbous' in text_lower:
            traits.append(ParsedTrait(
                category=TraitCategory.STEM,
                value='form: bulbous base',
                raw_text=text,
                confidence=0.8
            ))
        
        if 'ring' in text_lower or 'annulus' in text_lower:
            traits.append(ParsedTrait(
                category=TraitCategory.STEM,
                value='ring: present',
                raw_text=text,
                confidence=0.8
            ))
        
        if 'hollow' in text_lower:
            traits.append(ParsedTrait(
                category=TraitCategory.STEM,
                value='form: hollow',
                raw_text=text,
                confidence=0.8
            ))
        elif 'solid' in text_lower:
            traits.append(ParsedTrait(
                category=TraitCategory.STEM,
                value='form: solid',
                raw_text=text,
                confidence=0.8
            ))
        
        return traits
    
    def _extract_flesh_traits(self, text: str) -> List[ParsedTrait]:
        """Extract flesh-related traits."""
        traits = []
        text_lower = text.lower()
        
        # Color
        for color, patterns in self.COLOR_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                if any(word in text_lower for word in ['flesh', 'interior', 'inside']):
                    traits.append(ParsedTrait(
                        category=TraitCategory.FLESH,
                        value=f'color: {color}',
                        raw_text=text,
                        confidence=0.7
                    ))
                    break
        
        # Texture
        if 'firm' in text_lower:
            traits.append(ParsedTrait(
                category=TraitCategory.FLESH,
                value='texture: firm',
                raw_text=text,
                confidence=0.7
            ))
        elif 'soft' in text_lower:
            traits.append(ParsedTrait(
                category=TraitCategory.FLESH,
                value='texture: soft',
                raw_text=text,
                confidence=0.7
            ))
        
        return traits
    
    def _extract_habitat_traits(self, text: str, context: Optional[Dict[str, str]]) -> List[ParsedTrait]:
        """Extract habitat information."""
        traits = []
        text_lower = text.lower()
        
        # From context
        if context and 'habitat' in context:
            traits.append(ParsedTrait(
                category=TraitCategory.HABITAT,
                value=context['habitat'],
                raw_text=context['habitat'],
                confidence=0.9
            ))
            return traits
        
        # From text
        for habitat, patterns in self.HABITAT_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                traits.append(ParsedTrait(
                    category=TraitCategory.HABITAT,
                    value=habitat,
                    raw_text=text,
                    confidence=0.7
                ))
        
        # Tree associations
        for tree, patterns in self.TREE_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                traits.append(ParsedTrait(
                    category=TraitCategory.HABITAT,
                    value=f'under {tree}',
                    raw_text=text,
                    confidence=0.7
                ))
        
        return traits
    
    def _extract_season_traits(self, text: str, context: Optional[Dict[str, str]]) -> List[ParsedTrait]:
        """Extract season information."""
        traits = []
        text_lower = text.lower()
        
        # From context
        if context and 'season' in context:
            traits.append(ParsedTrait(
                category=TraitCategory.SEASON,
                value=context['season'],
                raw_text=context['season'],
                confidence=0.95
            ))
            return traits
        
        # From text
        for season, patterns in self.SEASON_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                traits.append(ParsedTrait(
                    category=TraitCategory.SEASON,
                    value=season,
                    raw_text=text,
                    confidence=0.8
                ))
                break
        
        return traits
    
    def _calculate_confidence(self, traits: List[ParsedTrait], identified_traits: Dict[str, str]) -> float:
        """Calculate overall confidence from extracted traits."""
        if not traits:
            return 0.0
        
        avg_confidence = sum(t.confidence for t in traits) / len(traits)
        
        # Reduce confidence if many required traits are missing
        identified_count = sum(1 for t in self.required_traits if t in identified_traits)
        trait_coverage = identified_count / len(self.required_traits)
        
        return avg_confidence * trait_coverage
    
    def _calculate_quality(self, identified_traits: Dict[str, str], ambiguous: List[str]) -> float:
        """Calculate observation quality (0-1)."""
        required_found = sum(1 for t in self.required_traits if t in identified_traits)
        total_found = len(identified_traits)
        
        required_ratio = required_found / len(self.required_traits)
        total_ratio = total_found / (len(self.required_traits) + len(self.optional_traits))
        
        # Penalize ambiguous traits
        ambiguous_penalty = len(ambiguous) * 0.05
        
        quality = (required_ratio * 0.6 + total_ratio * 0.4) - ambiguous_penalty
        return max(0.0, min(1.0, quality))
