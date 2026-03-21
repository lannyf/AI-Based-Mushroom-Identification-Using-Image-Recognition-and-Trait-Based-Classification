"""
Integration Tests for Hybrid Mushroom Classification System

Tests all three identification methods working together and validates:
- Confidence aggregation strategies
- Lookalike detection
- Safety warning systems
- End-to-end hybrid classification
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

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


def create_mock_predictions() -> Dict[str, Dict[str, Any]]:
    """
    Create mock predictions from all three methods for testing.
    
    Returns:
        Dict of test cases with method predictions
    """
    from models.hybrid_classifier import MethodPrediction
    
    return {
        'chanterelle_test': {
            'description': 'Yellow funnel-shaped mushroom with pale gills',
            'image': MethodPrediction(
                method='image',
                species='Chanterelle',
                confidence=0.92,
                reasoning='Yellow color and funnel shape match perfectly',
                top_k=[
                    ('Chanterelle', 0.92),
                    ('False Chanterelle', 0.05),
                    ('Pig\'s Ear', 0.03)
                ]
            ),
            'trait': MethodPrediction(
                method='trait',
                species='Chanterelle',
                confidence=0.85,
                reasoning='Cap shape, color, gill structure match training data',
                top_k=[
                    ('Chanterelle', 0.85),
                    ('Black Trumpet', 0.08),
                    ('Pig\'s Ear', 0.07)
                ]
            ),
            'llm': MethodPrediction(
                method='llm',
                species='Chanterelle',
                confidence=0.88,
                reasoning='Yellow funnel shape and decurrent ridges diagnostic',
                top_k=[
                    ('Chanterelle', 0.88),
                    ('Pig\'s Ear', 0.10),
                    ('False Chanterelle', 0.02)
                ]
            ),
            'expected': 'Chanterelle',
            'min_confidence': 0.85
        },
        'fly_agaric_test': {
            'description': 'Red cap with white spots, white gills, under birch',
            'image': MethodPrediction(
                method='image',
                species='Fly Agaric',
                confidence=0.95,
                reasoning='Red cap with white spots is diagnostic',
                top_k=[
                    ('Fly Agaric', 0.95),
                    ('Other Amanita', 0.05)
                ]
            ),
            'trait': MethodPrediction(
                method='trait',
                species='Fly Agaric',
                confidence=0.92,
                reasoning='Red cap, white spots, white gills, ring and volva all match',
                top_k=[
                    ('Fly Agaric', 0.92),
                    ('Amanita virosa', 0.08)
                ]
            ),
            'llm': MethodPrediction(
                method='llm',
                species='Fly Agaric',
                confidence=0.90,
                reasoning='Classic red with white spots under birch is Fly Agaric',
                top_k=[
                    ('Fly Agaric', 0.90),
                    ('Amanita virosa', 0.10)
                ]
            ),
            'expected': 'Fly Agaric',
            'min_confidence': 0.85,
            'expect_warning': True
        },
        'ambiguous_test': {
            'description': 'Yellow funnel-shaped but gills unclear',
            'image': MethodPrediction(
                method='image',
                species='Chanterelle',
                confidence=0.65,
                reasoning='Color and shape suggest Chanterelle but less certain',
                top_k=[
                    ('Chanterelle', 0.65),
                    ('False Chanterelle', 0.20),
                    ('Pig\'s Ear', 0.15)
                ]
            ),
            'trait': MethodPrediction(
                method='trait',
                species='False Chanterelle',
                confidence=0.58,
                reasoning='Traits could match either Chanterelle or False Chanterelle',
                top_k=[
                    ('False Chanterelle', 0.58),
                    ('Chanterelle', 0.35),
                    ('Pig\'s Ear', 0.07)
                ]
            ),
            'llm': MethodPrediction(
                method='llm',
                species='Chanterelle',
                confidence=0.72,
                reasoning='Probably Chanterelle but ambiguous gill description',
                top_k=[
                    ('Chanterelle', 0.72),
                    ('False Chanterelle', 0.20),
                    ('Pig\'s Ear', 0.08)
                ]
            ),
            'expected': 'Chanterelle',
            'min_confidence': 0.60,
            'expect_lookalike': 'False Chanterelle'
        }
    }


def test_aggregation_strategies():
    """Test different confidence aggregation strategies."""
    from models.hybrid_classifier import (
        HybridClassifier, AggregationMethod, MethodPrediction
    )
    
    logger.info('\n' + '='*60)
    logger.info('Testing Aggregation Strategies')
    logger.info('='*60)
    
    test_cases = create_mock_predictions()
    results = {'aggregation_tests': []}
    
    for strategy_name in [AggregationMethod.WEIGHTED_AVERAGE, 
                          AggregationMethod.GEOMETRIC_MEAN,
                          AggregationMethod.VOTING]:
        logger.info(f'\nTesting {strategy_name.value}:')
        
        classifier = HybridClassifier(aggregation_method=strategy_name)
        
        for test_id, test_data in test_cases.items():
            hybrid_result = classifier.classify(
                image_prediction=test_data['image'],
                trait_prediction=test_data['trait'],
                llm_prediction=test_data['llm']
            )
            
            passed = (
                hybrid_result.top_species.lower() in test_data['expected'].lower() and
                hybrid_result.top_confidence >= test_data['min_confidence']
            )
            
            results['aggregation_tests'].append({
                'strategy': strategy_name.value,
                'test_id': test_id,
                'expected': test_data['expected'],
                'predicted': hybrid_result.top_species,
                'confidence': hybrid_result.top_confidence,
                'consensus': hybrid_result.consensus_strength,
                'passed': passed
            })
            
            status = '✓' if passed else '✗'
            logger.info(f'  {status} {test_id}: {hybrid_result.top_species} ({hybrid_result.top_confidence:.2%})')
    
    return results


def test_lookalike_detection():
    """Test lookalike detection."""
    from models.hybrid_classifier import HybridClassifier, MethodPrediction
    
    logger.info('\n' + '='*60)
    logger.info('Testing Lookalike Detection')
    logger.info('='*60)
    
    classifier = HybridClassifier()
    test_cases = create_mock_predictions()
    results = {'lookalike_tests': []}
    
    for test_id, test_data in test_cases.items():
        if test_id == 'ambiguous_test':
            hybrid_result = classifier.classify(
                image_prediction=test_data['image'],
                trait_prediction=test_data['trait'],
                llm_prediction=test_data['llm']
            )
            
            lookalike_found = any(
                test_data.get('expect_lookalike', '').lower() in l[0].lower()
                for l in hybrid_result.lookalikes
            )
            
            results['lookalike_tests'].append({
                'test_id': test_id,
                'species': hybrid_result.top_species,
                'lookalikes_found': len(hybrid_result.lookalikes),
                'lookalike_list': [(l[0], l[1]) for l in hybrid_result.lookalikes],
                'expected_lookalike': test_data.get('expect_lookalike'),
                'found_expected': lookalike_found
            })
            
            logger.info(f'  {test_id}: Found {len(hybrid_result.lookalikes)} lookalike(s)')
            for species, similarity, reason in hybrid_result.lookalikes:
                logger.info(f'    - {species}: {similarity:.1%} ({reason})')
    
    return results


def test_safety_warnings():
    """Test safety warning generation."""
    from models.hybrid_classifier import HybridClassifier
    
    logger.info('\n' + '='*60)
    logger.info('Testing Safety Warnings')
    logger.info('='*60)
    
    classifier = HybridClassifier()
    test_cases = create_mock_predictions()
    results = {'safety_tests': []}
    
    for test_id, test_data in test_cases.items():
        hybrid_result = classifier.classify(
            image_prediction=test_data['image'],
            trait_prediction=test_data['trait'],
            llm_prediction=test_data['llm']
        )
        
        has_danger_warning = any('DANGER' in w for w in hybrid_result.safety_warnings)
        expect_warning = test_data.get('expect_warning', False)
        
        results['safety_tests'].append({
            'test_id': test_id,
            'species': hybrid_result.top_species,
            'warning_count': len(hybrid_result.safety_warnings),
            'danger_warning': has_danger_warning,
            'expected_warning': expect_warning,
            'warnings': hybrid_result.safety_warnings
        })
        
        logger.info(f'  {test_id} ({hybrid_result.top_species}):')
        for warning in hybrid_result.safety_warnings:
            logger.info(f'    {warning}')
    
    return results


def test_method_comparison():
    """Test method-by-method comparison."""
    from models.hybrid_classifier import HybridClassifier
    
    logger.info('\n' + '='*60)
    logger.info('Testing Method Comparison')
    logger.info('='*60)
    
    classifier = HybridClassifier()
    test_cases = create_mock_predictions()
    results = {'comparison_tests': []}
    
    for test_id, test_data in test_cases.items():
        predictions = {
            'image': test_data['image'],
            'trait': test_data['trait'],
            'llm': test_data['llm']
        }
        
        comparison = classifier.compare_methods(predictions)
        
        results['comparison_tests'].append({
            'test_id': test_id,
            'methods_used': comparison['methods_used'],
            'top_predictions': comparison['top_predictions'],
            'agreement': comparison['agreement'],
            'variance': comparison['variance']
        })
        
        logger.info(f'  {test_id}:')
        logger.info(f'    Consensus: {comparison["agreement"]["consensus"]}')
        logger.info(f'    Unique species: {comparison["agreement"]["species_count"]}')
        logger.info(f'    Confidence variance: {comparison["variance"]:.3f}')
        for method, pred in comparison['top_predictions'].items():
            logger.info(f'      {method}: {pred["species"]} ({pred["confidence"]:.2%})')
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Integration tests for hybrid classification system'
    )
    parser.add_argument(
        '--test',
        choices=['aggregation', 'lookalikes', 'safety', 'comparison', 'all'],
        default='all',
        help='Which test suite to run'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON'
    )
    
    args = parser.parse_args()
    
    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║  Hybrid Classification System - Integration Tests         ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    
    all_results = {}
    
    if args.test in ['aggregation', 'all']:
        all_results.update(test_aggregation_strategies())
    
    if args.test in ['lookalikes', 'all']:
        all_results.update(test_lookalike_detection())
    
    if args.test in ['safety', 'all']:
        all_results.update(test_safety_warnings())
    
    if args.test in ['comparison', 'all']:
        all_results.update(test_method_comparison())
    
    # Print summary
    logger.info('\n' + '='*60)
    logger.info('TEST SUMMARY')
    logger.info('='*60)
    
    if 'aggregation_tests' in all_results:
        passed = sum(1 for t in all_results['aggregation_tests'] if t['passed'])
        total = len(all_results['aggregation_tests'])
        logger.info(f'Aggregation tests: {passed}/{total} passed')
    
    if 'lookalike_tests' in all_results:
        logger.info(f'Lookalike tests: {len(all_results["lookalike_tests"])} completed')
    
    if 'safety_tests' in all_results:
        logger.info(f'Safety tests: {len(all_results["safety_tests"])} completed')
    
    if 'comparison_tests' in all_results:
        logger.info(f'Comparison tests: {len(all_results["comparison_tests"])} completed')
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f'\nResults saved to {output_path}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
