"""
ground_truth_generator.py - Generate/Validate Semantic Labels

Two modes:
1. GENERATE mode: Create labels using fuzzy inference (if no Semantic_Tag column)
2. VALIDATE mode: Compare existing Semantic_Tag with fuzzy inference results

Author: FuzSemCom Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.fuzzy_engine import create_fuzzy_system, get_semantic_output

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "data/raw/Agriculture_dataset_with_metadata.csv"
OUTPUT_FULL = "data/processed/semantic_dataset_fuzzy.csv"
OUTPUT_TRAIN = "data/processed/semantic_dataset_train.csv"
OUTPUT_TEST = "data/processed/semantic_dataset_test.csv"
STATS_FILE = "data/processed/fuzzy_generation_stats.txt"
VALIDATION_REPORT = "data/processed/validation_report.txt"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Valid ranges (from Table I in paper)
VALID_RANGES = {
    'soil_moisture': (0, 100),
    'pH': (4.0, 9.0),
    'nitrogen': (0, 300),
    'temperature': (10, 40),
    'humidity': (30, 100)
}

# Column mapping for different naming conventions
COLUMN_MAP = {
    'Moisture': 'soil_moisture',
    'pH': 'pH',
    'N': 'nitrogen',
    'P': 'phosphorus',
    'K': 'potassium',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'NDI_Label': 'ndi_label',
    'PDI_Label': 'pdi_label',
    'Semantic_Tag': 'semantic_tag_original'
}


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(input_path: str) -> pd.DataFrame:
    """
    Load and preprocess raw dataset.
    
    Args:
        input_path: Path to raw CSV file
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\n" + "="*80)
    print("LOADING & PREPROCESSING DATA")
    print("="*80)
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} samples")
    
    # Apply column mapping
    print("\nApplying column mapping...")
    for old, new in COLUMN_MAP.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
            print(f"  ✓ {old} → {new}")
    
    # Check required columns
    required_cols = ['soil_moisture', 'pH', 'nitrogen', 'temperature', 'humidity']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")
    
    print(f"\n✓ All required input columns present: {required_cols}")
    
    # Check if dataset already has semantic labels
    has_existing_labels = 'semantic_tag_original' in df.columns
    if has_existing_labels:
        print(f"✓ Dataset contains existing semantic labels (Semantic_Tag)")
        print(f"  Mode: VALIDATION (compare with fuzzy inference)")
    else:
        print(f"⚠️  No existing semantic labels found")
        print(f"  Mode: GENERATION (create labels using fuzzy inference)")
    
    return df, has_existing_labels


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean input data.
    
    Args:
        df: Raw dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "="*80)
    print("DATA VALIDATION & CLEANING")
    print("="*80)
    
    original_len = len(df)
    
    # Remove rows with missing values in required columns
    required_cols = ['soil_moisture', 'pH', 'nitrogen', 'temperature', 'humidity']
    df_clean = df.dropna(subset=required_cols).copy()
    removed_na = original_len - len(df_clean)
    if removed_na > 0:
        print(f"✓ Removed {removed_na:,} rows with missing values")
    
    # Validate ranges
    invalid_rows = set()
    for col, (min_val, max_val) in VALID_RANGES.items():
        mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
        invalid_count = mask.sum()
        if invalid_count > 0:
            print(f"⚠️  {invalid_count:,} rows with {col} out of range [{min_val}, {max_val}]")
            invalid_rows.update(df_clean[mask].index.tolist())
    
    # Remove invalid rows
    if invalid_rows:
        df_clean = df_clean.drop(list(invalid_rows)).reset_index(drop=True)
        print(f"✓ Removed {len(invalid_rows):,} rows with out-of-range values")
    
    print(f"\n✓ Final dataset: {len(df_clean):,} valid samples")
    print(f"  Removed: {original_len - len(df_clean):,} samples ({(original_len - len(df_clean))/original_len*100:.1f}%)")
    
    return df_clean


# ============================================================================
# FUZZY INFERENCE
# ============================================================================

def generate_fuzzy_labels(df: pd.DataFrame) -> tuple:
    """
    Generate semantic labels using fuzzy inference.
    
    Args:
        df: Cleaned dataframe with sensor readings
    
    Returns:
        tuple: (labels list, confidences list)
    """
    print("\n" + "="*80)
    print("FUZZY INFERENCE")
    print("="*80)
    
    # Create fuzzy system (expert-defined MFs, no training)
    sim, out, antecedents, rules = create_fuzzy_system()
    print("✓ Fuzzy system initialized (expert-defined membership functions)")
    
    labels = []
    confidences = []
    errors = 0
    
    print(f"\nProcessing {len(df):,} samples...")
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0 and idx > 0:
            print(f"  Progress: {idx:,}/{len(df):,} ({idx/len(df)*100:.1f}%)")
        
        try:
            # Prepare inputs (all 5 variables)
            inputs = {
                'soil_moisture': float(row['soil_moisture']),
                'pH': float(row['pH']),
                'nitrogen': float(row['nitrogen']),
                'temperature': float(row['temperature']),
                'humidity': float(row['humidity'])
            }
            
            # Get semantic output with confidence
            semantic_label, confidence = get_semantic_output(
                sim, inputs, antecedents, rules
            )
            
            labels.append(semantic_label)
            confidences.append(confidence)
            
        except Exception as e:
            # Fallback for inference errors
            labels.append('other')
            confidences.append(0)
            errors += 1
    
    print(f"✓ Generated labels for {len(df):,} samples")
    if errors > 0:
        print(f"⚠️  {errors:,} inference errors (fallback to 'other')")
    
    return labels, confidences


# ============================================================================
# VALIDATION (if existing labels present)
# ============================================================================

def validate_against_existing_labels(df: pd.DataFrame) -> dict:
    """
    Compare fuzzy inference results with existing Semantic_Tag.
    
    Args:
        df: Dataframe with both semantic_label_fuzzy and semantic_tag_original
    
    Returns:
        dict: Validation metrics
    """
    print("\n" + "="*80)
    print("VALIDATION: Fuzzy Inference vs Existing Labels")
    print("="*80)
    
    # Get labels
    fuzzy_labels = df['semantic_label_fuzzy'].tolist()
    original_labels = df['semantic_tag_original'].tolist()
    
    # Calculate accuracy
    accuracy = accuracy_score(original_labels, fuzzy_labels)
    
    # Confusion matrix
    unique_labels = sorted(set(original_labels + fuzzy_labels))
    cm = confusion_matrix(original_labels, fuzzy_labels, labels=unique_labels)
    
    # Classification report
    report = classification_report(original_labels, fuzzy_labels, 
                                   output_dict=True, zero_division=0)
    
    # Print results
    print(f"\n✓ Validation Accuracy: {accuracy*100:.2f}%")
    print(f"  (Agreement between fuzzy inference and existing labels)")
    
    print(f"\nLabel Distribution Comparison:")
    print(f"{'Label':<30} {'Original':<15} {'Fuzzy':<15} {'Diff':<10}")
    print("-"*70)
    
    orig_counts = df['semantic_tag_original'].value_counts()
    fuzzy_counts = df['semantic_label_fuzzy'].value_counts()
    
    all_labels = sorted(set(orig_counts.index) | set(fuzzy_counts.index))
    for label in all_labels:
        orig_cnt = orig_counts.get(label, 0)
        fuzzy_cnt = fuzzy_counts.get(label, 0)
        diff = fuzzy_cnt - orig_cnt
        print(f"{label:<30} {orig_cnt:>6,} ({orig_cnt/len(df)*100:5.1f}%)  "
              f"{fuzzy_cnt:>6,} ({fuzzy_cnt/len(df)*100:5.1f}%)  "
              f"{diff:>+6,}")
    
    validation_results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'label_distribution_original': orig_counts.to_dict(),
        'label_distribution_fuzzy': fuzzy_counts.to_dict()
    }
    
    return validation_results


# ============================================================================
# STATISTICS
# ============================================================================

def print_statistics(df: pd.DataFrame, label_col: str = 'semantic_label_fuzzy'):
    """Print label distribution and statistics."""
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION")
    print("="*80)
    
    label_counts = df[label_col].value_counts()
    label_percentages = (label_counts / len(df) * 100).round(1)
    
    for label, count in label_counts.items():
        pct = label_percentages[label]
        bar = "█" * int(pct / 2)
        print(f"{label:30s} {count:6,} ({pct:5.1f}%) {bar}")
    
    if 'confidence' in df.columns:
        print("\n" + "="*80)
        print("CONFIDENCE STATISTICS")
        print("="*80)
        print(f"Mean Confidence:    {df['confidence'].mean():.1f}/255")
        print(f"Median Confidence:  {df['confidence'].median():.1f}/255")
        print(f"Min Confidence:     {df['confidence'].min()}/255")
        print(f"Max Confidence:     {df['confidence'].max()}/255")
        print(f"Std Confidence:     {df['confidence'].std():.1f}")
    
    print("="*80 + "\n")


def save_statistics(df: pd.DataFrame, output_path: str, 
                   validation_results: dict = None):
    """Save statistics to text file."""
    with open(output_path, 'w') as f:
        f.write("FuzSemCom Ground Truth Generation Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Samples: {len(df):,}\n\n")
        
        # Fuzzy inference distribution
        f.write("Fuzzy Inference Label Distribution:\n")
        f.write("-"*80 + "\n")
        label_counts = df['semantic_label_fuzzy'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            f.write(f"{label:30s} {count:6,} ({pct:5.1f}%)\n")
        
        # Confidence statistics
        if 'confidence' in df.columns:
            f.write("\n" + "="*80 + "\n")
            f.write("Confidence Statistics:\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean:   {df['confidence'].mean():.1f}/255\n")
            f.write(f"Median: {df['confidence'].median():.1f}/255\n")
            f.write(f"Min:    {df['confidence'].min()}/255\n")
            f.write(f"Max:    {df['confidence'].max()}/255\n")
            f.write(f"Std:    {df['confidence'].std():.1f}\n")
        
        # Validation results (if available)
        if validation_results:
            f.write("\n" + "="*80 + "\n")
            f.write("Validation Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy: {validation_results['accuracy']*100:.2f}%\n")
            f.write("(Agreement between fuzzy inference and existing labels)\n")
    
    print(f"✓ Statistics saved to {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_ground_truth(
    input_csv_path: str = INPUT_CSV,
    output_full_path: str = OUTPUT_FULL,
    output_train_path: str = OUTPUT_TRAIN,
    output_test_path: str = OUTPUT_TEST,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Main pipeline to generate/validate semantic labels.
    
    Steps:
    1. Load and preprocess data
    2. Validate and clean data
    3. Generate fuzzy inference labels
    4. Compare with existing labels (if available)
    5. Save datasets
    6. Generate statistics
    """
    print("\n" + "="*80)
    print("FUZSEMCOM GROUND TRUTH GENERATION PIPELINE")
    print("="*80)
    print(f"Input:  {input_csv_path}")
    print(f"Output: {output_full_path}")
    print("")
    
    # Step 1: Load data
    df, has_existing_labels = load_and_preprocess_data(input_csv_path)
    
    # Step 2: Validate and clean
    df_clean = validate_and_clean_data(df)
    
    # Step 3: Generate fuzzy labels
    labels, confidences = generate_fuzzy_labels(df_clean)
    df_clean['semantic_label_fuzzy'] = labels
    df_clean['confidence'] = confidences
    
    # Step 4: Validation (if existing labels present)
    validation_results = None
    if has_existing_labels:
        validation_results = validate_against_existing_labels(df_clean)
    
    # Step 5: Determine final label column
    if has_existing_labels:
        # Use original labels as ground truth, keep fuzzy for comparison
        df_clean['semantic_label'] = df_clean['semantic_tag_original']
        print(f"\n✓ Using original Semantic_Tag as ground truth")
        print(f"  (Fuzzy inference labels saved as 'semantic_label_fuzzy')")
    else:
        # Use fuzzy labels as ground truth
        df_clean['semantic_label'] = df_clean['semantic_label_fuzzy']
        print(f"\n✓ Using fuzzy inference as ground truth")
    
    # Step 6: Save full dataset
    print(f"\n{'='*80}")
    print("SAVING DATASETS")
    print("="*80)
    
    df_clean.to_csv(output_full_path, index=False)
    print(f"✓ Full dataset: {output_full_path}")
    
    # Step 7: Split train/test
    print(f"\nSplitting train/test ({int((1-test_size)*100)}/{int(test_size*100)})...")
    
    # Stratified split based on final semantic_label
    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean['semantic_label']
    )
    
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    print(f"✓ Train: {len(train_df):,} samples → {output_train_path}")
    print(f"✓ Test:  {len(test_df):,} samples → {output_test_path}")
    
    # Step 8: Generate statistics
    print_statistics(df_clean, 'semantic_label_fuzzy')
    save_statistics(df_clean, STATS_FILE, validation_results)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    
    if validation_results:
        print(f"\nValidation Accuracy: {validation_results['accuracy']*100:.2f}%")
        print("(See detailed report in data/processed/validation_report.txt)")
    
    print("")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    generate_ground_truth(
        input_csv_path=INPUT_CSV,
        output_full_path=OUTPUT_FULL,
        output_train_path=OUTPUT_TRAIN,
        output_test_path=OUTPUT_TEST,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
