"""
04_evaluate_fse.py - Comprehensive Evaluation of Fuzzy Semantic Encoder

Evaluates FuzSemCom's fuzzy inference system on test dataset:
- Semantic accuracy (overall + per-class)
- Confidence score analysis
- Confusion matrix visualization
- Per-class precision/recall/F1
- Symbol encoding statistics
- Comparison with expected results from paper

Author: FuzSemCom Team
Date: 2025-11-16
"""

import sys
from pathlib import Path
import os
import json
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)

from src.fuzzy_engine import (
    create_fuzzy_system, 
    get_semantic_output,
    SYMBOL_CODES
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
TEST_DATASET = ROOT / "data/processed/semantic_dataset_test.csv"
FULL_DATASET = ROOT / "data/processed/semantic_dataset_fuzzy.csv"  # Fallback

RESULTS_DIR = ROOT / "results/reports"
FIGURES_DIR = ROOT / "results/figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Expected results from paper (Section IV.D)
PAPER_EXPECTED_ACCURACY = 0.887  # 88.7%

# Label order for confusion matrix (from most to least frequent)
LABEL_ORDER = [
    'optimal',
    'water_deficit_acidic',
    'water_deficit_alkaline',
    'nutrient_deficiency',
    'fungal_risk',
    'acidic_soil',
    'alkaline_soil',
    'heat_stress',
    'other'
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data():
    """
    Load test dataset for evaluation.
    
    Returns:
        pd.DataFrame: Test dataset
    """
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    # Try test split first
    if TEST_DATASET.exists():
        df = pd.read_csv(TEST_DATASET)
        print(f"✓ Loaded test dataset: {TEST_DATASET}")
        print(f"  Samples: {len(df):,}")
    elif FULL_DATASET.exists():
        df = pd.read_csv(FULL_DATASET)
        print(f⚠️  Test split not found, using full dataset: {FULL_DATASET}")
        print(f"  Samples: {len(df):,}")
    else:
        raise FileNotFoundError(
            f"Neither test dataset ({TEST_DATASET}) nor "
            f"full dataset ({FULL_DATASET}) found. "
            f"Please run ground_truth_generator.py first."
        )
    
    # Validate required columns
    required_cols = ['soil_moisture', 'pH', 'nitrogen', 'temperature', 
                     'humidity', 'semantic_label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    print(f"\n✓ Label distribution:")
    for label, count in df['semantic_label'].value_counts().items():
        print(f"    {label:30s}: {count:5,} ({count/len(df)*100:5.1f}%)")
    
    return df


# ============================================================================
# FUZZY INFERENCE & PREDICTION
# ============================================================================

def run_inference(df: pd.DataFrame):
    """
    Run fuzzy inference on test dataset.
    
    Args:
        df: Test dataframe
    
    Returns:
        tuple: (predictions, confidences, errors)
    """
    print("\n" + "="*80)
    print("RUNNING FUZZY INFERENCE")
    print("="*80)
    
    # Create fuzzy system
    sim, out, antecedents, rules = create_fuzzy_system()
    print("✓ Fuzzy system initialized")
    
    predictions = []
    confidences = []
    errors = 0
    
    print(f"\nProcessing {len(df):,} samples...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0 and idx > 0:
            print(f"  Progress: {idx:,}/{len(df):,} ({idx/len(df)*100:.1f}%)")
        
        inputs = {
            'soil_moisture': float(row['soil_moisture']),
            'pH': float(row['pH']),
            'nitrogen': float(row['nitrogen']),
            'temperature': float(row['temperature']),
            'humidity': float(row['humidity'])
        }
        
        try:
            pred_label, confidence = get_semantic_output(
                sim, inputs, antecedents, rules
            )
            predictions.append(pred_label)
            confidences.append(confidence)
        except Exception as e:
            # Fallback for inference errors
            predictions.append('other')
            confidences.append(0)
            errors += 1
    
    print(f"✓ Inference complete")
    if errors > 0:
        print(f"⚠️  {errors:,} inference errors ({errors/len(df)*100:.2f}%)")
    
    return predictions, confidences, errors


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(true_labels, predictions, confidences):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        true_labels: Ground truth labels
        predictions: Predicted labels
        confidences: Confidence scores
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Per-class metrics
    report = classification_report(
        true_labels, 
        predictions, 
        output_dict=True,
        zero_division=0,
        labels=LABEL_ORDER
    )
    
    # Confusion matrix
    cm = confusion_matrix(
        true_labels, 
        predictions,
        labels=LABEL_ORDER
    )
    
    # Precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0
    )
    
    # Confidence statistics
    conf_stats = {
        'mean': np.mean(confidences),
        'median': np.median(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'q25': np.percentile(confidences, 25),
        'q75': np.percentile(confidences, 75)
    }
    
    # Symbol encoding statistics
    symbol_counts = {}
    for pred in predictions:
        symbol = SYMBOL_CODES.get(pred, 0xFF)
        symbol_counts[f"0x{symbol:02X}"] = symbol_counts.get(f"0x{symbol:02X}", 0) + 1
    
    metrics = {
        'accuracy': accuracy,
        'per_class_accuracy': {
            label: report[label]['precision'] 
            for label in LABEL_ORDER if label in report
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'labels': LABEL_ORDER,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        },
        'confidence_statistics': conf_stats,
        'symbol_encoding_distribution': symbol_counts,
        'paper_comparison': {
            'expected_accuracy': PAPER_EXPECTED_ACCURACY,
            'actual_accuracy': accuracy,
            'difference': accuracy - PAPER_EXPECTED_ACCURACY,
            'difference_percent': (accuracy - PAPER_EXPECTED_ACCURACY) * 100
        }
    }
    
    # Print summary
    print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
    print(f"  Expected (paper): {PAPER_EXPECTED_ACCURACY*100:.2f}%")
    print(f"  Difference:       {(accuracy - PAPER_EXPECTED_ACCURACY)*100:+.2f}%")
    
    print(f"\n✓ Confidence Statistics:")
    print(f"  Mean:   {conf_stats['mean']:.1f}/255 ({conf_stats['mean']/255*100:.1f}%)")
    print(f"  Median: {conf_stats['median']:.1f}/255")
    print(f"  Std:    {conf_stats['std']:.1f}")
    print(f"  Range:  [{conf_stats['min']}, {conf_stats['max']}]")
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, labels, output_path):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        labels: Label names
        output_path: Save path
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {output_path}")
    plt.close()


def plot_per_class_metrics(metrics, output_path):
    """
    Plot per-class precision, recall, F1 scores.
    
    Args:
        metrics: Metrics dictionary
        output_path: Save path
    """
    pcm = metrics['per_class_metrics']
    labels = pcm['labels']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    axes[0].barh(labels, pcm['precision'], color='skyblue')
    axes[0].set_xlabel('Precision')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_xlim([0, 1])
    axes[0].grid(axis='x', alpha=0.3)
    
    # Recall
    axes[1].barh(labels, pcm['recall'], color='lightcoral')
    axes[1].set_xlabel('Recall')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_xlim([0, 1])
    axes[1].grid(axis='x', alpha=0.3)
    
    # F1 Score
    axes[2].barh(labels, pcm['f1_score'], color='lightgreen')
    axes[2].set_xlabel('F1 Score')
    axes[2].set_title('Per-Class F1 Score')
    axes[2].set_xlim([0, 1])
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-class metrics to {output_path}")
    plt.close()


def plot_confidence_distribution(confidences, output_path):
    """
    Plot confidence score distribution.
    
    Args:
        confidences: List of confidence scores
        output_path: Save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(confidences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(confidences):.1f}')
    axes[0].axvline(np.median(confidences), color='green', linestyle='--',
                    label=f'Median: {np.median(confidences):.1f}')
    axes[0].set_xlabel('Confidence Score (0-255)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence Score Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(confidences, vert=True)
    axes[1].set_ylabel('Confidence Score (0-255)')
    axes[1].set_title('Confidence Score Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confidence distribution to {output_path}")
    plt.close()


def plot_symbol_encoding_distribution(symbol_counts, output_path):
    """
    Plot symbol encoding distribution.
    
    Args:
        symbol_counts: Dictionary of symbol counts
        output_path: Save path
    """
    symbols = list(symbol_counts.keys())
    counts = list(symbol_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(symbols, counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Symbol Code (Hex)')
    plt.ylabel('Frequency')
    plt.title('Symbol Encoding Distribution')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved symbol distribution to {output_path}")
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_text_report(metrics, output_path):
    """
    Generate detailed text report.
    
    Args:
        metrics: Metrics dictionary
        output_path: Save path
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FUZZY SEMANTIC ENCODER EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write("1. OVERALL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Semantic Accuracy:        {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Expected (from paper):    {PAPER_EXPECTED_ACCURACY*100:.2f}%\n")
        f.write(f"Difference:               {metrics['paper_comparison']['difference_percent']:+.2f}%\n")
        f.write("\n")
        
        # Confidence statistics
        conf = metrics['confidence_statistics']
        f.write("2. CONFIDENCE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean:       {conf['mean']:.1f}/255 ({conf['mean']/255*100:.1f}%)\n")
        f.write(f"Median:     {conf['median']:.1f}/255\n")
        f.write(f"Std Dev:    {conf['std']:.1f}\n")
        f.write(f"Min:        {conf['min']}/255\n")
        f.write(f"Max:        {conf['max']}/255\n")
        f.write(f"Q1:         {conf['q25']:.1f}/255\n")
        f.write(f"Q3:         {conf['q75']:.1f}/255\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("3. PER-CLASS PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Label':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        pcm = metrics['per_class_metrics']
        for i, label in enumerate(pcm['labels']):
            f.write(f"{label:<30} "
                   f"{pcm['precision'][i]:<12.4f} "
                   f"{pcm['recall'][i]:<12.4f} "
                   f"{pcm['f1_score'][i]:<12.4f} "
                   f"{pcm['support'][i]:<10}\n")
        f.write("\n")
        
        # Symbol encoding
        f.write("4. SYMBOL ENCODING DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for symbol, count in sorted(metrics['symbol_encoding_distribution'].items()):
            pct = count / sum(metrics['symbol_encoding_distribution'].values()) * 100
            f.write(f"{symbol}  {count:6,} ({pct:5.1f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved text report to {output_path}")


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_fse():
    """
    Main evaluation pipeline.
    
    Steps:
    1. Load test data
    2. Run fuzzy inference
    3. Calculate metrics
    4. Generate visualizations
    5. Save results
    """
    print("\n" + "="*80)
    print("FUZZY SEMANTIC ENCODER EVALUATION")
    print("="*80)
    
    # Step 1: Load data
    df = load_test_data()
    
    # Step 2: Run inference
    predictions, confidences, errors = run_inference(df)
    
    # Step 3: Calculate metrics
    true_labels = df['semantic_label'].tolist()
    metrics = calculate_metrics(true_labels, predictions, confidences)
    
    # Step 4: Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        LABEL_ORDER,
        FIGURES_DIR / "fse_confusion_matrix.png"
    )
    
    plot_per_class_metrics(
        metrics,
        FIGURES_DIR / "fse_per_class_metrics.png"
    )
    
    plot_confidence_distribution(
        confidences,
        FIGURES_DIR / "fse_confidence_distribution.png"
    )
    
    plot_symbol_encoding_distribution(
        metrics['symbol_encoding_distribution'],
        FIGURES_DIR / "fse_symbol_distribution.png"
    )
    
    # Step 5: Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # JSON results
    json_path = RESULTS_DIR / "fse_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Saved JSON results to {json_path}")
    
    # Text report
    report_path = RESULTS_DIR / "fse_evaluation_report.txt"
    generate_text_report(metrics, report_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Overall Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Expected (paper):     {PAPER_EXPECTED_ACCURACY*100:.2f}%")
    print(f"Difference:           {metrics['paper_comparison']['difference_percent']:+.2f}%")
    print(f"Mean Confidence:      {metrics['confidence_statistics']['mean']:.1f}/255")
    print(f"Inference Errors:     {errors:,}")
    print("\nOutput files:")
    print(f"  - JSON:             {json_path}")
    print(f"  - Report:           {report_path}")
    print(f"  - Figures:          {FIGURES_DIR}")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    evaluate_fse()
