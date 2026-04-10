"""
Evaluate Trait-Based Classification Model

Comprehensive evaluation including confusion matrices, per-class metrics,
feature importance, and comparative analysis between algorithms.

Usage:
    python scripts/evaluate_trait_model.py --algorithm random_forest
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from models.trait_processor import TraitDataset
from models.trait_classifier import TraitClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_model(algorithm: str, artifacts_dir: str) -> Tuple[TraitClassifier, TraitDataset]:
    """Load a trained model and dataset."""
    model_path = os.path.join(artifacts_dir, f'trait_classifier_{algorithm}.pkl')
    scaler_path = os.path.join(artifacts_dir, f'trait_scaler_{algorithm}.pkl')
    metadata_path = os.path.join(artifacts_dir, f'trait_metadata_{algorithm}.pkl')
    
    # Create classifier instance
    classifier = TraitClassifier(algorithm=algorithm, n_species=20)
    classifier.load(model_path, scaler_path, metadata_path)
    
    # Load dataset for reference
    dataset = TraitDataset(
        os.path.join(project_root, 'data/raw/species_traits.xml'),
        os.path.join(project_root, 'data/raw/species.csv')
    )
    
    return classifier, dataset


def evaluate_comprehensive(classifier: TraitClassifier, 
                          dataset: TraitDataset,
                          artifacts_dir: str = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of trained model.
    
    Args:
        classifier: Trained TraitClassifier
        dataset: TraitDataset for reference
        artifacts_dir: Directory to save evaluation results
    
    Returns:
        Dictionary with detailed evaluation results
    """
    
    if artifacts_dir is None:
        artifacts_dir = os.path.join(project_root, 'artifacts')
    
    logger.info("=" * 70)
    logger.info("TRAIT CLASSIFIER EVALUATION")
    logger.info("=" * 70)
    
    # Prepare data
    X, y, feature_names = dataset.prepare_features()
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    val_ratio = 0.15 / (0.7 + 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    # Evaluate on each set
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    
    train_results = classifier.evaluate(X_train, y_train)
    val_results = classifier.evaluate(X_val, y_val)
    test_results = classifier.evaluate(X_test, y_test)
    
    # Log results
    logger.info("\nTRAINING SET:")
    logger.info(f"  Accuracy:  {train_results['accuracy']:.4f}")
    logger.info(f"  Precision: {train_results['precision']:.4f}")
    logger.info(f"  Recall:    {train_results['recall']:.4f}")
    logger.info(f"  F1-Score:  {train_results['f1']:.4f}")
    
    logger.info("\nVALIDATION SET:")
    logger.info(f"  Accuracy:  {val_results['accuracy']:.4f}")
    logger.info(f"  Precision: {val_results['precision']:.4f}")
    logger.info(f"  Recall:    {val_results['recall']:.4f}")
    logger.info(f"  F1-Score:  {val_results['f1']:.4f}")
    
    logger.info("\nTEST SET:")
    logger.info(f"  Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall:    {test_results['recall']:.4f}")
    logger.info(f"  F1-Score:  {test_results['f1']:.4f}")
    
    # Feature importance
    logger.info("\n" + "=" * 70)
    logger.info("TOP 15 MOST IMPORTANT FEATURES")
    logger.info("=" * 70)
    
    try:
        feature_importance = classifier.get_feature_importance(top_n=15)
        for i, (feature_name, importance) in enumerate(feature_importance, 1):
            logger.info(f"{i:2d}. {feature_name:40s} {importance:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Per-class metrics
    logger.info("\n" + "=" * 70)
    logger.info("PER-CLASS METRICS (Test Set)")
    logger.info("=" * 70)
    
    per_class = test_results['classification_report']
    for class_name in classifier.class_names:
        if class_name in per_class:
            metrics = per_class[class_name]
            logger.info(f"\n{class_name}:")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score:  {metrics.get('f1-score', 0):.4f}")
    
    # Create confusion matrix visualization
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classifier.class_names,
                yticklabels=classifier.class_names)
    plt.title(f'Confusion Matrix - {classifier.algorithm.replace("_", " ").title()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(artifacts_dir, f'trait_confusion_matrix_{classifier.algorithm}.png')
    plt.savefig(cm_path, dpi=150)
    logger.info(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    # Plot feature importance
    try:
        feature_importance = classifier.get_feature_importance(top_n=15)
        features, importances = zip(*feature_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top 15 Features - {classifier.algorithm.replace("_", " ").title()}')
        plt.tight_layout()
        
        fi_path = os.path.join(artifacts_dir, f'trait_feature_importance_{classifier.algorithm}.png')
        plt.savefig(fi_path, dpi=150)
        logger.info(f"Feature importance plot saved to {fi_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {e}")
    
    # Save evaluation results to JSON
    eval_results = {
        'algorithm': classifier.algorithm,
        'train_metrics': {
            'accuracy': float(train_results['accuracy']),
            'precision': float(train_results['precision']),
            'recall': float(train_results['recall']),
            'f1': float(train_results['f1'])
        },
        'val_metrics': {
            'accuracy': float(val_results['accuracy']),
            'precision': float(val_results['precision']),
            'recall': float(val_results['recall']),
            'f1': float(val_results['f1'])
        },
        'test_metrics': {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1']),
            'roc_auc': float(test_results['roc_auc'])
        },
        'feature_importance': dict(
            classifier.get_feature_importance(top_n=15)
        ) if hasattr(classifier.model, 'feature_importances_') else {}
    }
    
    eval_path = os.path.join(artifacts_dir, f'trait_evaluation_{classifier.algorithm}.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    
    return eval_results


def compare_algorithms(artifacts_dir: str = None) -> None:
    """
    Compare performance of different algorithms.
    
    Args:
        artifacts_dir: Directory with saved results
    """
    
    if artifacts_dir is None:
        artifacts_dir = os.path.join(project_root, 'artifacts')
    
    logger.info("\n" + "=" * 70)
    logger.info("ALGORITHM COMPARISON")
    logger.info("=" * 70)
    
    results = {}
    
    for algorithm in ['decision_tree', 'random_forest']:
        results_file = os.path.join(artifacts_dir, f'trait_evaluation_{algorithm}.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results[algorithm] = json.load(f)
    
    if len(results) > 1:
        # Create comparison table
        print("\n" + "=" * 70)
        print("ALGORITHM PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"\n{'Metric':<15} {'Decision Tree':<20} {'Random Forest':<20}")
        print("-" * 55)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in metrics:
            dt_val = results['decision_tree']['test_metrics'].get(metric, 0)
            rf_val = results['random_forest']['test_metrics'].get(metric, 0)
            
            print(f"{metric.capitalize():<15} {dt_val:>19.4f} {rf_val:>19.4f}")
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics comparison
        algorithms = list(results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        for i, algo in enumerate(algorithms):
            values = [results[algo]['test_metrics'].get(m, 0) for m in metrics_names]
            axes[0].bar(x + i*width, values, width, label=algo.replace('_', ' ').title())
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Algorithm Performance Comparison')
        axes[0].set_xticks(x + width/2)
        axes[0].set_xticklabels(metrics_names)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # ROC AUC comparison
        roc_scores = [results[algo]['test_metrics'].get('roc_auc', 0) for algo in algorithms]
        axes[1].bar(algorithms, roc_scores, color=['skyblue', 'lightcoral'])
        axes[1].set_ylabel('ROC AUC')
        axes[1].set_title('ROC AUC Score Comparison')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(axis='y', alpha=0.3)
        
        # Format x-axis labels
        axes[1].set_xticklabels([a.replace('_', ' ').title() for a in algorithms])
        
        plt.tight_layout()
        comp_path = os.path.join(artifacts_dir, 'trait_algorithm_comparison.png')
        plt.savefig(comp_path, dpi=150)
        logger.info(f"Comparison chart saved to {comp_path}")
        plt.close()
    else:
        logger.info("Only one algorithm found. Skipping comparison.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate trait-based mushroom classification model'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['decision_tree', 'random_forest', 'all'],
        default='all',
        help='Algorithm to evaluate'
    )
    
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default=None,
        help='Directory with saved model artifacts'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all algorithms'
    )
    
    args = parser.parse_args()
    
    artifacts_dir = args.artifacts_dir or os.path.join(project_root, 'artifacts')
    
    # Evaluate specified algorithms
    if args.algorithm == 'all':
        algorithms = ['decision_tree', 'random_forest']
    else:
        algorithms = [args.algorithm]
    
    for algo in algorithms:
        try:
            classifier, dataset = load_trained_model(algo, artifacts_dir)
            evaluate_comprehensive(classifier, dataset, artifacts_dir)
        except FileNotFoundError as e:
            logger.error(f"Model not found for {algo}: {e}")
            logger.info(f"Train the model first with: python scripts/train_trait_model.py --algorithm {algo}")
    
    # Compare algorithms if requested or if evaluating all
    if args.compare or args.algorithm == 'all':
        compare_algorithms(artifacts_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
