"""
Hybrid Mushroom Classification System

Combines predictions from all three identification methods:
1. Image Recognition (CNN transfer learning)
2. Trait-Based Classification (Decision Tree / Random Forest)
3. LLM-Based Classification (Natural language processing)

Implements multiple confidence aggregation strategies and lookalike detection.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Available confidence aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    VOTING = "voting"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


@dataclass
class MethodPrediction:
    """Single prediction from one identification method."""
    method: str  # 'image', 'trait', 'llm'
    species: str
    confidence: float
    reasoning: str = ""
    top_k: List[Tuple[str, float]] = field(default_factory=list)  # Top k predictions


@dataclass
class HybridResult:
    """Final hybrid classification result."""
    top_species: str
    top_confidence: float
    predictions: List[Tuple[str, float, str]]  # (species, confidence, method_agreement)
    method_predictions: Dict[str, MethodPrediction]  # Per-method results
    aggregation_method: str
    safety_warnings: List[str]
    lookalikes: List[Tuple[str, float, str]]  # (species, similarity, reason)
    confidence_breakdown: Dict[str, float]  # Per-method confidence for top species
    consensus_strength: float  # How much agreement between methods (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'top_species': self.top_species,
            'confidence': self.top_confidence,
            'predictions': [
                {'species': p[0], 'confidence': p[1], 'method_agreement': p[2]}
                for p in self.predictions
            ],
            'method_predictions': {
                method: {
                    'species': pred.species,
                    'confidence': pred.confidence,
                    'reasoning': pred.reasoning
                }
                for method, pred in self.method_predictions.items()
            },
            'aggregation_method': self.aggregation_method,
            'safety_warnings': self.safety_warnings,
            'lookalikes': [
                {'species': l[0], 'similarity': l[1], 'reason': l[2]}
                for l in self.lookalikes
            ],
            'confidence_breakdown': self.confidence_breakdown,
            'consensus_strength': self.consensus_strength
        }


class AggregationStrategy(ABC):
    """Abstract base class for confidence aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, predictions: Dict[str, MethodPrediction], 
                 top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Aggregate predictions from multiple methods.
        
        Args:
            predictions: Dict of method_name -> MethodPrediction
            top_k: Return top k predictions
        
        Returns:
            List of (species, aggregated_confidence) tuples, sorted by confidence
        """
        pass


class WeightedAverageStrategy(AggregationStrategy):
    """Weighted average of confidences across methods."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with optional custom weights.
        
        Args:
            weights: Dict of method -> weight (default: equal weights)
        """
        self.weights = weights or {
            'image': 0.4,  # Higher weight for image (most direct)
            'trait': 0.35,  # Trait-based also important
            'llm': 0.25    # LLM less reliable on unknowns
        }
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def aggregate(self, predictions: Dict[str, MethodPrediction], 
                 top_k: int = 5) -> List[Tuple[str, float]]:
        """Aggregate using weighted average."""
        species_scores = {}
        
        for method, pred in predictions.items():
            weight = self.weights.get(method, 0.33)
            
            # Add confidence for top prediction
            if pred.species not in species_scores:
                species_scores[pred.species] = 0.0
            species_scores[pred.species] += weight * pred.confidence
            
            # Add weighted scores for top-k predictions
            for species, conf in pred.top_k:
                if species != pred.species:
                    if species not in species_scores:
                        species_scores[species] = 0.0
                    # Reduce weight for non-top predictions
                    species_scores[species] += (weight * 0.5) * conf
        
        # Sort and return top k
        sorted_species = sorted(species_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_species[:top_k]


class GeometricMeanStrategy(AggregationStrategy):
    """Geometric mean of confidences (better for probability combination)."""
    
    def aggregate(self, predictions: Dict[str, MethodPrediction], 
                 top_k: int = 5) -> List[Tuple[str, float]]:
        """Aggregate using geometric mean."""
        species_scores = {}
        species_counts = {}
        
        for method, pred in predictions.items():
            # Geometric mean requires all values > 0, so use 0.01 as minimum
            conf = max(0.01, pred.confidence)
            
            if pred.species not in species_scores:
                species_scores[pred.species] = 1.0
                species_counts[pred.species] = 0
            
            species_scores[pred.species] *= conf
            species_counts[pred.species] += 1
            
            # Process top-k
            for species, c in pred.top_k:
                c = max(0.01, c)
                if species != pred.species:
                    if species not in species_scores:
                        species_scores[species] = 1.0
                        species_counts[species] = 0
                    species_scores[species] *= (c ** 0.5)  # Reduce influence
                    species_counts[species] += 1
        
        # Calculate geometric mean
        final_scores = {}
        for species, product in species_scores.items():
            count = species_counts[species]
            if count > 0:
                geo_mean = product ** (1.0 / count)
                final_scores[species] = geo_mean
        
        sorted_species = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_species[:top_k]


class VotingStrategy(AggregationStrategy):
    """Voting-based aggregation (ranked voting)."""
    
    def aggregate(self, predictions: Dict[str, MethodPrediction], 
                 top_k: int = 5) -> List[Tuple[str, float]]:
        """Aggregate using ranked voting."""
        species_votes = {}
        
        for method, pred in predictions.items():
            # Give points based on ranking
            points = {pred.species: len(predictions) * 10}  # Top prediction gets max points
            
            for rank, (species, conf) in enumerate(pred.top_k[:10]):
                if species not in points:
                    points[species] = 0
                points[species] += (10 - rank) * conf
            
            # Accumulate votes
            for species, score in points.items():
                if species not in species_votes:
                    species_votes[species] = 0.0
                species_votes[species] += score
        
        # Normalize to confidence range [0, 1]
        if species_votes:
            max_votes = max(species_votes.values())
            species_votes = {s: v/max_votes for s, v in species_votes.items()}
        
        sorted_species = sorted(species_votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_species[:top_k]


class LookalikeMatcher:
    """Detects visually/morphologically similar species (lookalikes)."""
    
    # Hardcoded lookalike pairs (would normally come from CSV)
    LOOKALIKE_PAIRS = [
        ('Chanterelle', 'False Chanterelle', 0.8, 'Similar funnel shape and yellow color'),
        ('Chanterelle', "Pig's Ear", 0.7, 'Both funnel-shaped with ridges'),
        ('Black Trumpet', 'Chanterelle', 0.6, 'Both funnel-shaped but different colors'),
        ('Fly Agaric', 'Amanita virosa', 0.9, 'Both deadly Amanita species; confusion by inexperienced foragers can be very dangerous'),
        ('Porcini', 'Other Boletus', 0.8, 'Similar pore structure'),
        ('Slippery Jack', 'Other Boletus', 0.7, 'Slimy cap, pores underneath'),
    ]
    
    def __init__(self):
        """Initialize lookalike database."""
        self.lookalikes = self.LOOKALIKE_PAIRS
    
    def find_lookalikes(self, species: str, other_predictions: List[Tuple[str, float]]) -> List[Tuple[str, float, str]]:
        """
        Find lookalike species from the predictions.
        
        Args:
            species: Target species
            other_predictions: List of (species, confidence) from hybrid predictions
        
        Returns:
            List of (lookalike_species, similarity_score, reason)
        """
        result = []
        other_species_set = {s[0] for s in other_predictions}
        
        for sp1, sp2, similarity, reason in self.LOOKALIKE_PAIRS:
            if (sp1.lower() == species.lower() and sp2 in other_species_set):
                result.append((sp2, similarity, reason))
            elif (sp2.lower() == species.lower() and sp1 in other_species_set):
                result.append((sp1, similarity, reason))
        
        return sorted(result, key=lambda x: x[1], reverse=True)


class SafetySystem:
    """Manages safety warnings and multi-method confirmation."""
    
    # Hardcoded safety data (would normally come from database)
    TOXIC_SPECIES = {
        'Fly Agaric': 'TOXIC - Contains psychoactive compounds (ibotenic acid, muscimol)',
        'Amanita virosa': 'DEADLY - Destroys liver and kidneys (amatoxins)',
        'Amanita phalloides': 'DEADLY - Death cap (amatoxins)',
    }
    
    EDIBLE_SPECIES = {
        'Chanterelle': 'SAFE - Well-known edible mushroom',
        'Porcini': 'SAFE - Prized culinary mushroom',
        'Black Trumpet': 'SAFE - Delicious wild mushroom',
        "Pig's Ear": 'SAFE - Edible when cooked',
        'Slippery Jack': 'SAFE - Popular for cooking',
        'Giant Puffball': 'SAFE - Young stages edible',
    }
    
    def __init__(self):
        """Initialize safety database."""
        self.toxic = self.TOXIC_SPECIES
        self.edible = self.EDIBLE_SPECIES
    
    def get_warnings(self, species: str, confidence_breakdown: Dict[str, float]) -> List[str]:
        """
        Generate safety warnings based on species and confidence.
        
        Args:
            species: Identified species
            confidence_breakdown: Per-method confidences
        
        Returns:
            List of safety warnings
        """
        warnings = []
        
        # Check if toxic
        if species in self.toxic:
            warnings.append(f"⚠️  DANGER: {self.toxic[species]}")
            
            # Check for multi-method confirmation
            method_count = sum(1 for conf in confidence_breakdown.values() if conf > 0.5)
            if method_count < 2:
                warnings.append(f"⚠️  WARNING: Only {method_count} method(s) confirm this identification. "
                              "Get expert verification before handling!")
        
        # Check if edible but low confidence
        elif species in self.edible:
            avg_confidence = sum(confidence_breakdown.values()) / len(confidence_breakdown)
            if avg_confidence < 0.5:
                warnings.append(f"⚠️  DANGEROUS: Very low confidence ({avg_confidence:.0%}). "
                              "Do NOT consume without expert verification!")
            elif avg_confidence < 0.7:
                warnings.append(f"⚠️  WARNING: Identification confidence is only {avg_confidence:.0%}. "
                              "Verify with expert before consuming.")
        else:
            # Unknown species
            warnings.append("⚠️  WARNING: Unknown species. Please verify with expert mycologist.")
        
        # General disclaimer
        warnings.append("⚠️  DISCLAIMER: This system is for educational purposes only. "
                       "Never use it as sole basis for edibility determination.")
        
        return warnings


class HybridClassifier:
    """Main hybrid classification engine combining all three methods."""
    
    def __init__(self, aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid classifier.
        
        Args:
            aggregation_method: Strategy for combining predictions
            weights: Custom weights for weighted averaging (if used)
        """
        self.aggregation_method = aggregation_method
        
        # Initialize aggregation strategy
        if aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            self.strategy = WeightedAverageStrategy(weights)
        elif aggregation_method == AggregationMethod.GEOMETRIC_MEAN:
            self.strategy = GeometricMeanStrategy()
        elif aggregation_method == AggregationMethod.VOTING:
            self.strategy = VotingStrategy()
        else:
            raise ValueError(f'Unknown aggregation method: {aggregation_method}')
        
        self.lookalike_matcher = LookalikeMatcher()
        self.safety_system = SafetySystem()
        
        logger.info(f'HybridClassifier initialized with {aggregation_method.value} aggregation')
    
    def classify(self, image_prediction: Optional[MethodPrediction] = None,
                trait_prediction: Optional[MethodPrediction] = None,
                llm_prediction: Optional[MethodPrediction] = None) -> HybridResult:
        """
        Perform hybrid classification combining all available predictions.
        
        Args:
            image_prediction: Result from Image CNN (Phase 2)
            trait_prediction: Result from Trait Classifier (Phase 3)
            llm_prediction: Result from LLM Classifier (Phase 4)
        
        Returns:
            HybridResult with aggregated predictions and analysis
        """
        if not any([image_prediction, trait_prediction, llm_prediction]):
            raise ValueError('At least one prediction method required')
        
        # Collect predictions
        predictions = {}
        if image_prediction:
            predictions['image'] = image_prediction
        if trait_prediction:
            predictions['trait'] = trait_prediction
        if llm_prediction:
            predictions['llm'] = llm_prediction
        
        logger.info(f'Aggregating predictions from {len(predictions)} method(s)')
        
        # Aggregate confidences
        aggregated = self.strategy.aggregate(predictions, top_k=10)
        
        if not aggregated:
            raise ValueError('Aggregation produced no results')
        
        # Get top prediction
        top_species, top_confidence = aggregated[0]
        top_5 = aggregated[:5]
        
        # Calculate confidence breakdown for top species
        confidence_breakdown = {
            method: pred.confidence if pred.species == top_species else 
                   next((c for s, c in pred.top_k if s == top_species), 0.0)
            for method, pred in predictions.items()
        }
        
        # Calculate consensus strength (agreement between methods)
        consensus = self._calculate_consensus(top_species, predictions)
        
        # Find lookalikes
        lookalikes = self.lookalike_matcher.find_lookalikes(top_species, top_5)
        
        # Get safety warnings
        warnings = self.safety_system.get_warnings(top_species, confidence_breakdown)
        
        # Format predictions with method agreement
        formatted_predictions = [
            (species, confidence, self._get_method_agreement_label(species, predictions))
            for species, confidence in top_5
        ]
        
        return HybridResult(
            top_species=top_species,
            top_confidence=top_confidence,
            predictions=formatted_predictions,
            method_predictions=predictions,
            aggregation_method=self.aggregation_method.value,
            safety_warnings=warnings,
            lookalikes=lookalikes,
            confidence_breakdown=confidence_breakdown,
            consensus_strength=consensus
        )
    
    def _calculate_consensus(self, species: str, predictions: Dict[str, MethodPrediction]) -> float:
        """
        Calculate how much agreement there is between methods.
        
        Returns:
            Consensus strength 0-1 (1 = all methods agree)
        """
        if len(predictions) < 2:
            return 1.0  # Single method is "unanimous"
        
        # Count how many methods agree on this species as top prediction
        agreement = sum(1 for pred in predictions.values() if pred.species == species)
        total = len(predictions)
        
        return agreement / total
    
    def _get_method_agreement_label(self, species: str, predictions: Dict[str, MethodPrediction]) -> str:
        """Get human-readable label for method agreement."""
        agreeing_methods = [method for method, pred in predictions.items() if pred.species == species]
        
        if len(agreeing_methods) == len(predictions):
            return 'All methods agree'
        elif len(agreeing_methods) >= len(predictions) * 0.66:
            return 'Majority agreement'
        elif len(agreeing_methods) > 0:
            return f'{len(agreeing_methods)}/{len(predictions)} methods'
        else:
            return 'Lower ranked by all'
    
    def compare_methods(self, predictions: Dict[str, MethodPrediction]) -> Dict[str, Any]:
        """
        Compare predictions from individual methods.
        
        Returns:
            Comparison analysis
        """
        comparison = {
            'methods_used': list(predictions.keys()),
            'top_predictions': {},
            'agreement': {},
            'variance': 0.0
        }
        
        for method, pred in predictions.items():
            comparison['top_predictions'][method] = {
                'species': pred.species,
                'confidence': pred.confidence
            }
        
        # Calculate agreement
        species_list = [pred.species for pred in predictions.values()]
        unique_species = set(species_list)
        comparison['agreement']['consensus'] = len(unique_species) == 1
        comparison['agreement']['species_count'] = len(unique_species)
        
        # Calculate variance in confidence
        confidences = [pred.confidence for pred in predictions.values()]
        if len(confidences) > 1:
            comparison['variance'] = statistics.stdev(confidences)
        
        return comparison
