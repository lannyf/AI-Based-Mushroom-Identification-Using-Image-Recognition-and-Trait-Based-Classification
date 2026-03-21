"""
Train/Validate LLM-Based Classifier

This script validates the LLM classifier on test cases and measures performance
against known species. Includes end-to-end testing of:
- LLM response parsing
- Confidence calibration
- Species prediction accuracy
- Safety warning identification
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Setup logging
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


def load_test_cases() -> List[Dict[str, Any]]:
    """Load test cases for validation."""
    return [
        {
            'id': 'test_01',
            'observation': 'Yellow mushroom with funnel-shaped cap and pale gills. Found on forest floor in mixed woods.',
            'expected_species': 'Chanterelle',
            'min_confidence': 0.7,
            'toxicity': 'SAFE'
        },
        {
            'id': 'test_02',
            'observation': 'Red cap with white spots, white gills, white stem with ring and bulbous base under birch tree in autumn.',
            'expected_species': 'Fly Agaric',
            'min_confidence': 0.8,
            'toxicity': 'TOXIC'
        },
        {
            'id': 'test_03',
            'observation': 'Brown convex cap with yellow pores underneath (not true gills). Pale stem with network pattern.',
            'expected_species': 'Porcini',
            'min_confidence': 0.7,
            'toxicity': 'SAFE'
        },
        {
            'id': 'test_04',
            'observation': 'Dark gray to black funnel-shaped mushroom with pale yellowish ridges, found in forest in summer.',
            'expected_species': 'Black Trumpet',
            'min_confidence': 0.7,
            'toxicity': 'SAFE'
        },
        {
            'id': 'test_05',
            'observation': 'Brown funnel-shaped mushroom, similar to chanterelle but darker. Pale yellow ridges, hollow stem.',
            'expected_species': "Pig's Ear",
            'min_confidence': 0.6,
            'toxicity': 'SAFE'
        },
        {
            'id': 'test_06',
            'observation': 'Small white mushroom with red cap and white spots, slimy coating, under birch in autumn.',
            'expected_species': 'Fly Agaric',
            'min_confidence': 0.7,
            'toxicity': 'TOXIC'
        },
        {
            'id': 'test_07',
            'observation': 'Spherical white puffball mushroom in meadow during summer.',
            'expected_species': 'Giant Puffball',
            'min_confidence': 0.6,
            'toxicity': 'SAFE'
        },
        {
            'id': 'test_08',
            'observation': 'Yellow-brown slimy cap with yellow pores, growing under pine trees.',
            'expected_species': 'Slippery Jack',
            'min_confidence': 0.6,
            'toxicity': 'SAFE'
        },
    ]


def validate_classifier(backend_type: str = 'mock', api_key: str = None) -> Dict[str, Any]:
    """
    Validate LLM classifier on test cases.
    
    Args:
        backend_type: 'mock', 'openai', or other backend
        api_key: API key if required by backend
    
    Returns:
        Validation results dictionary
    """
    from models.llm_classifier import LLMClassifier
    from models.observation_parser import ObservationParser
    
    logger.info(f'Initializing LLM classifier with {backend_type} backend')
    
    try:
        classifier = LLMClassifier(backend_type=backend_type, api_key=api_key)
        parser = ObservationParser()
    except Exception as e:
        logger.error(f'Failed to initialize classifier: {e}')
        return {'status': 'error', 'message': str(e)}
    
    test_cases = load_test_cases()
    results = {
        'backend': backend_type,
        'total_tests': len(test_cases),
        'passed': 0,
        'failed': 0,
        'accuracy': 0.0,
        'test_results': []
    }
    
    logger.info(f'Running {len(test_cases)} test cases...')
    
    for test in test_cases:
        logger.info(f"\nTest {test['id']}: {test['observation'][:50]}...")
        
        # Parse observation
        parsed = parser.parse(test['observation'])
        logger.info(f'  Parsed traits: {parsed.identified_traits}')
        logger.info(f'  Quality score: {parsed.quality_score:.2f}')
        
        # Classify
        prediction = classifier.classify(test['observation'])
        
        # Check if prediction matches expected
        top_species_match = prediction.top_species.lower() in test['expected_species'].lower() or \
                           test['expected_species'].lower() in prediction.top_species.lower()
        
        confidence_ok = prediction.top_confidence >= test['min_confidence']
        
        # Check safety warnings for toxic species
        is_toxic = test['toxicity'] == 'TOXIC'
        has_warning = len(prediction.safety_warnings) > 0
        safety_ok = (is_toxic and has_warning) or (not is_toxic and not has_warning)
        
        test_passed = top_species_match and confidence_ok and safety_ok
        
        result = {
            'test_id': test['id'],
            'observation': test['observation'],
            'expected': test['expected_species'],
            'predicted': prediction.top_species,
            'confidence': prediction.top_confidence,
            'species_match': top_species_match,
            'confidence_ok': confidence_ok,
            'safety_ok': safety_ok,
            'passed': test_passed,
            'reasoning': prediction.reasoning,
            'safety_warnings': prediction.safety_warnings,
            'processing_time_ms': prediction.processing_time_ms
        }
        
        results['test_results'].append(result)
        
        if test_passed:
            results['passed'] += 1
            logger.info(f'  ✓ PASSED')
        else:
            results['failed'] += 1
            logger.info(f'  ✗ FAILED')
            if not top_species_match:
                logger.info(f'    - Species mismatch: expected {test["expected_species"]}, got {prediction.top_species}')
            if not confidence_ok:
                logger.info(f'    - Low confidence: {prediction.top_confidence:.2f} < {test["min_confidence"]}')
            if not safety_ok:
                logger.info(f'    - Safety warning issue: expected={is_toxic}, found={has_warning}')
    
    results['accuracy'] = results['passed'] / results['total_tests']
    
    return results


def test_observation_parser() -> Dict[str, Any]:
    """Test the observation parser."""
    from models.observation_parser import ObservationParser
    
    logger.info('Testing ObservationParser...')
    parser = ObservationParser()
    
    test_observations = [
        'Yellow mushroom with funnel-shaped cap and pale gills.',
        'Red cap with white spots, white gills, growing under birch.',
        'Brown convex cap with pores, found in pine forest.',
    ]
    
    results = {'parser_tests': []}
    
    for obs in test_observations:
        parsed = parser.parse(obs)
        results['parser_tests'].append({
            'observation': obs,
            'traits_found': parsed.identified_traits,
            'quality': parsed.quality_score,
            'missing': parsed.missing_traits
        })
        logger.info(f'  ✓ Parsed: {obs[:40]}... (quality: {parsed.quality_score:.2f})')
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate LLM-Based Mushroom Classifier'
    )
    parser.add_argument(
        '--backend',
        choices=['mock', 'openai'],
        default='mock',
        help='LLM backend to use (default: mock)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for LLM backend (if required)'
    )
    parser.add_argument(
        '--test-parser',
        action='store_true',
        help='Run observation parser tests only'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    logger.info('╔═══════════════════════════════════════════════════════════╗')
    logger.info('║  LLM-Based Classifier Validation                          ║')
    logger.info('╚═══════════════════════════════════════════════════════════╝')
    logger.info(f'Backend: {args.backend}')
    
    all_results = {}
    
    # Test observation parser
    parser_results = test_observation_parser()
    all_results.update(parser_results)
    
    # Test classifier (unless --test-parser only)
    if not args.test_parser:
        classifier_results = validate_classifier(args.backend, args.api_key)
        all_results.update(classifier_results)
        
        logger.info('\n' + '='*60)
        logger.info('SUMMARY')
        logger.info('='*60)
        logger.info(f"Tests passed: {classifier_results['passed']}/{classifier_results['total_tests']}")
        logger.info(f"Accuracy: {classifier_results['accuracy']:.1%}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f'\nResults saved to {output_path}')
    
    return 0 if all_results.get('accuracy', 1.0) > 0.5 else 1


if __name__ == '__main__':
    sys.exit(main())
