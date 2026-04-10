"""
Train Trait-Based Classification Model

Trains decision tree and random forest models on mushroom trait data.
Provides hyperparameter options and comprehensive evaluation metrics.

Usage:
    python scripts/train_trait_model.py --algorithm random_forest --epochs 10
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from models.trait_processor import TraitEncoder, TraitDataset
from models.trait_classifier import TraitClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_trait_model(
    traits_csv: str = None,
    species_csv: str = None,
    algorithm: str = 'random_forest',
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    artifacts_dir: str = None
) -> Tuple[TraitClassifier, dict]:
    """
    Train trait-based classification model.
    
    Args:
        traits_csv: Path to species_traits.csv
        species_csv: Path to species.csv
        algorithm: Model algorithm ('decision_tree' or 'random_forest')
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed
        artifacts_dir: Directory to save artifacts
    
    Returns:
        Trained classifier and results dictionary
    """
    
    # Set default paths
    if traits_csv is None:
        traits_csv = os.path.join(project_root, 'data/raw/species_traits.xml')
    if species_csv is None:
        species_csv = os.path.join(project_root, 'data/raw/species.csv')
    if artifacts_dir is None:
        artifacts_dir = os.path.join(project_root, 'artifacts')
    
    # Create artifacts directory
    Path(artifacts_dir).mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("TRAIT-BASED MUSHROOM CLASSIFICATION TRAINING")
    logger.info("=" * 70)
    logger.info(f"Traits CSV: {traits_csv}")
    logger.info(f"Species CSV: {species_csv}")
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Train/Val/Test split: {train_size}/{val_size}/{test_size}")
    
    # Load and prepare data
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    dataset = TraitDataset(traits_csv, species_csv)
    X, y, feature_names = dataset.prepare_features()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of classes: {len(np.unique(y))}")
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Feature names (first 10): {feature_names[:10]}")
    
    # Split data: train/val/test
    logger.info("\n" + "=" * 70)
    logger.info("SPLITTING DATA")
    logger.info("=" * 70)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Val set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Get species names for class labels
    species_names = [
        dataset.get_species_name(i) for i in range(len(np.unique(y)))
    ]
    logger.info(f"Species classes: {species_names}")
    
    # Initialize and train classifier
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL")
    logger.info("=" * 70)
    
    classifier = TraitClassifier(
        algorithm=algorithm,
        n_species=len(np.unique(y)),
        random_state=random_state
    )
    
    train_metrics = classifier.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        feature_names=feature_names,
        class_names=species_names
    )
    
    logger.info(f"Train accuracy: {train_metrics['train_accuracy']:.4f}")
    if 'val_accuracy' in train_metrics:
        logger.info(f"Val accuracy: {train_metrics['val_accuracy']:.4f}")
    
    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 70)
    
    test_results = classifier.evaluate(X_test, y_test)
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_results['precision']:.4f}")
    logger.info(f"Test Recall: {test_results['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_results['f1']:.4f}")
    
    # Feature importance
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 70)
    
    try:
        feature_importance = classifier.get_feature_importance(top_n=10)
        for i, (feature_name, importance) in enumerate(feature_importance, 1):
            logger.info(f"{i:2d}. {feature_name:40s} {importance:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Sample predictions
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 70)
    
    sample_predictions = classifier.predict_with_confidence(X_test[:3], top_k=3)
    for i, predictions in enumerate(sample_predictions):
        true_label = dataset.get_species_name(y_test[i])
        logger.info(f"\nSample {i+1} (True: {true_label})")
        for species, confidence in predictions:
            logger.info(f"  - {species}: {confidence:.4f}")
    
    # Save model
    logger.info("\n" + "=" * 70)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 70)
    
    model_path = os.path.join(artifacts_dir, f'trait_classifier_{algorithm}.pkl')
    scaler_path = os.path.join(artifacts_dir, f'trait_scaler_{algorithm}.pkl')
    metadata_path = os.path.join(artifacts_dir, f'trait_metadata_{algorithm}.pkl')
    encoder_path = os.path.join(artifacts_dir, 'trait_encoder.pkl')
    
    classifier.save(model_path, scaler_path, metadata_path)
    dataset.save_encoder(encoder_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Encoder saved to {encoder_path}")
    
    # Save results to JSON
    results = {
        'algorithm': algorithm,
        'train_metrics': train_metrics,
        'test_metrics': {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1']),
            'roc_auc': float(test_results['roc_auc'])
        },
        'dataset': {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_classes': int(len(np.unique(y))),
            'train_size': int(X_train.shape[0]),
            'val_size': int(X_val.shape[0]),
            'test_size': int(X_test.shape[0])
        }
    }
    
    results_path = os.path.join(artifacts_dir, f'trait_results_{algorithm}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    return classifier, results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train trait-based mushroom classification model'
    )
    
    parser.add_argument(
        '--traits-csv',
        type=str,
        default=None,
        help='Path to species_traits.xml'
    )
    
    parser.add_argument(
        '--species-csv',
        type=str,
        default=None,
        help='Path to species.csv'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['decision_tree', 'random_forest'],
        default='random_forest',
        help='Classification algorithm to use'
    )
    
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.7,
        help='Proportion of data for training'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Proportion of data for validation'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.15,
        help='Proportion of data for testing'
    )
    
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default=None,
        help='Directory to save model artifacts'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Train model
    classifier, results = train_trait_model(
        traits_csv=args.traits_csv,
        species_csv=args.species_csv,
        algorithm=args.algorithm,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
