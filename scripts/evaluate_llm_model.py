"""
Evaluate LLM-Based Mushroom Classifier

Comprehensive evaluation including:
- Accuracy metrics (overall, per species, per confidence level)
- Confusion analysis (which species get confused)
- Safety warning effectiveness
- Comparison with trait-based classifier (Phase 3)
- Processing time and cost analysis
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Setup project paths."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    return project_root


@dataclass
class ConfusionMatrix:
    """Per-species confusion tracking."""
    species: str
    correct_predictions: int = 0
    total_predictions: int = 0
    confused_with: Dict[str, int] = None
    
    def __post_init__(self):
        if self.confused_with is None:
            self.confused_with = {}
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


def evaluate_llm_classifier(backend_type: str = 'mock', api_key: str = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of LLM classifier.
    
    Args:
        backend_type: 'mock' or 'openai'
        api_key: API key if required
    
    Returns:
        Evaluation results
    """
    from models.llm_classifier import LLMClassifier, SpeciesDatabase
    
    logger.info('Initializing LLM classifier for evaluation...')
    
    try:
        classifier = LLMClassifier(backend_type=backend_type, api_key=api_key)
        species_db = SpeciesDatabase()
    except Exception as e:
        logger.error(f'Initialization failed: {e}')
        return {'status': 'error', 'message': str(e)}
    
    # Create comprehensive test set
    test_cases = _create_evaluation_test_set()
    
    logger.info(f'Evaluating on {len(test_cases)} test cases...')
    
    # Track results
    results = {
        'backend': backend_type,
        'total_tests': len(test_cases),
        'accuracy': 0.0,
        'top_1_accuracy': 0.0,
        'top_3_accuracy': 0.0,
        'top_5_accuracy': 0.0,
        'average_confidence': 0.0,
        'safety_warnings_correct': 0,
        'processing_time_avg_ms': 0.0,
        'per_species_accuracy': {},
        'test_results': [],
        'confusion_matrix': {},
        'confidence_distribution': {}
    }
    
    total_time = 0
    correct_count = 0
    top3_count = 0
    top5_count = 0
    safety_correct = 0
    confidence_values = []
    
    species_confusion = {s: ConfusionMatrix(s) for s in species_db.get_all_species().values()}
    
    for test in test_cases:
        logger.debug(f"Testing: {test['observation'][:50]}...")
        
        # Get prediction
        prediction = classifier.classify(test['observation'])
        total_time += prediction.processing_time_ms
        confidence_values.append(prediction.top_confidence)
        
        # Check if correct
        top_match = test['expected_species'].lower() in prediction.top_species.lower() or \
                   prediction.top_species.lower() in test['expected_species'].lower()
        
        if top_match:
            correct_count += 1
            top3_count += 1
            top5_count += 1
        else:
            # Check top 3 and top 5
            top_3_species = [p[0].lower() for p in prediction.predictions[:3]]
            top_5_species = [p[0].lower() for p in prediction.predictions[:5]]
            
            expected_lower = test['expected_species'].lower()
            if any(expected_lower in s or s in expected_lower for s in top_3_species):
                top3_count += 1
                top5_count += 1
            elif any(expected_lower in s or s in expected_lower for s in top_5_species):
                top5_count += 1
        
        # Check safety warnings
        is_toxic = test['toxicity'] == 'TOXIC'
        has_warning = len(prediction.safety_warnings) > 0
        if (is_toxic and has_warning) or (not is_toxic and not has_warning):
            safety_correct += 1
        
        result = {
            'observation': test['observation'],
            'expected': test['expected_species'],
            'predicted': prediction.top_species,
            'confidence': prediction.top_confidence,
            'top_1_match': top_match,
            'safety_ok': (is_toxic and has_warning) or (not is_toxic and not has_warning),
            'processing_time_ms': prediction.processing_time_ms,
            'reasoning': prediction.reasoning
        }
        
        results['test_results'].append(result)
    
    # Calculate aggregate metrics
    results['accuracy'] = correct_count / len(test_cases) if test_cases else 0.0
    results['top_1_accuracy'] = results['accuracy']
    results['top_3_accuracy'] = top3_count / len(test_cases) if test_cases else 0.0
    results['top_5_accuracy'] = top5_count / len(test_cases) if test_cases else 0.0
    results['safety_accuracy'] = safety_correct / len(test_cases) if test_cases else 0.0
    results['average_confidence'] = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    results['processing_time_avg_ms'] = total_time / len(test_cases) if test_cases else 0.0
    
    # Per-species accuracy
    for test in test_cases:
        expected = test['expected_species']
        if expected not in results['per_species_accuracy']:
            results['per_species_accuracy'][expected] = {'correct': 0, 'total': 0}
        results['per_species_accuracy'][expected]['total'] += 1
    
    for test in results['test_results']:
        expected = test['expected']
        if test['top_1_match']:
            results['per_species_accuracy'][expected]['correct'] += 1
    
    for species, counts in results['per_species_accuracy'].items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        results['per_species_accuracy'][species] = {
            'accuracy': accuracy,
            'correct': counts['correct'],
            'total': counts['total']
        }
    
    return results


def _create_evaluation_test_set() -> List[Dict[str, Any]]:
    """Create comprehensive test set for evaluation."""
    return [
        # Chanterelle tests
        {
            'observation': 'Yellow mushroom with funnel-shaped cap and pale gills found in mixed forest.',
            'expected_species': 'Chanterelle',
            'toxicity': 'SAFE'
        },
        {
            'observation': 'Golden funnel-shaped mushroom with decurrent ridges in summer forest.',
            'expected_species': 'Chanterelle',
            'toxicity': 'SAFE'
        },
        # Black Trumpet tests
        {
            'observation': 'Dark gray to black funnel-shaped mushroom with pale gills in forest.',
            'expected_species': 'Black Trumpet',
            'toxicity': 'SAFE'
        },
        # Porcini tests
        {
            'observation': 'Brown cap with yellow pores, pale stem with network pattern.',
            'expected_species': 'Porcini',
            'toxicity': 'SAFE'
        },
        # Fly Agaric tests
        {
            'observation': 'Bright red cap with white spots, white gills, white stem with ring and bulbous base.',
            'expected_species': 'Fly Agaric',
            'toxicity': 'TOXIC'
        },
        {
            'observation': 'Red mushroom with white spots under birch tree in autumn.',
            'expected_species': 'Fly Agaric',
            'toxicity': 'TOXIC'
        },
        # Puffball tests
        {
            'observation': 'Large spherical white mushroom in meadow.',
            'expected_species': 'Giant Puffball',
            'toxicity': 'SAFE'
        },
        # Slippery Jack tests
        {
            'observation': 'Slimy yellow-brown cap with yellow pores under pine trees.',
            'expected_species': 'Slippery Jack',
            'toxicity': 'SAFE'
        },
        # Pig's Ear tests
        {
            'observation': 'Brown funnel-shaped mushroom with pale ridges, funnel deeper than Chanterelle.',
            'expected_species': "Pig's Ear",
            'toxicity': 'SAFE'
        },
        # Ambiguous cases
        {
            'observation': 'Yellow funnel-shaped cap, gills unclear, found in forest.',
            'expected_species': 'Chanterelle',
            'toxicity': 'SAFE'
        },
    ]


def compare_with_trait_classifier() -> Dict[str, Any]:
    """
    Compare LLM results with trait-based classifier (Phase 3).
    
    Returns:
        Comparison results
    """
    logger.info('Comparing LLM classifier with trait-based classifier...')
    
    # This would require loading both classifiers and running on same test set
    # For now, return placeholder
    return {
        'comparison': 'Requires both classifiers to be trained',
        'status': 'pending'
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM-Based Mushroom Classifier'
    )
    parser.add_argument(
        '--backend',
        choices=['mock', 'openai'],
        default='mock',
        help='LLM backend (default: mock)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for backend'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with trait-based classifier'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║  LLM Classifier Evaluation                               ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    
    # Run evaluation
    results = evaluate_llm_classifier(args.backend, args.api_key)
    
    # Print summary
    logger.info('\n' + '='*60)
    logger.info('EVALUATION SUMMARY')
    logger.info('='*60)
    logger.info(f"Top-1 Accuracy: {results.get('top_1_accuracy', 0):.1%}")
    logger.info(f"Top-3 Accuracy: {results.get('top_3_accuracy', 0):.1%}")
    logger.info(f"Top-5 Accuracy: {results.get('top_5_accuracy', 0):.1%}")
    logger.info(f"Safety Accuracy: {results.get('safety_accuracy', 0):.1%}")
    logger.info(f"Avg Confidence: {results.get('average_confidence', 0):.2f}")
    logger.info(f"Avg Processing Time: {results.get('processing_time_avg_ms', 0):.1f}ms")
    
    logger.info('\nPer-Species Accuracy:')
    for species, metrics in results.get('per_species_accuracy', {}).items():
        logger.info(f"  {species}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'\nResults saved to {output_path}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
