"""
Dataset Preprocessing for FuzSemCom
Maps expert labels (NDI/PDI) to semantic symbols following Section IV.A.1
Author: FuzSemCom Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "Agriculture_dataset_with_metadata.csv"
OUTPUT_FILE = "processed_data.csv"
STATS_FILE = "preprocessing_stats.txt"

REQUIRED_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
EXPERT_LABEL_COLUMNS = ["NDI_Label", "PDI_Label"]  # Expert annotations

# Semantic symbol encoding (1-byte hex codes)
SYMBOL_MAP = {
    'optimal': 0x00,
    'water_deficit_acidic': 0x02,
    'water_deficit_alkaline': 0x03,
    'acidic_soil': 0x04,
    'alkaline_soil': 0x05,
    'heat_stress': 0x06,
    'nutrient_deficiency': 0x07,
    'fungal_risk': 0x08,
    'ambiguous': 0xFF  # Will be filtered out
}

# Agronomic thresholds (from Table I in paper)
THRESHOLDS = {
    'moisture': {'low': 30, 'high': 60},
    'ph': {'acidic': 5.8, 'alkaline': 7.5, 'ideal_low': 6.0, 'ideal_high': 6.8},
    'nitrogen': {'low': 50, 'high': 100},
    'temperature': {'cool': 22, 'hot': 30, 'ideal_low': 22, 'ideal_high': 26},
    'humidity': {'dry': 60, 'humid': 80, 'ideal_low': 60, 'ideal_high': 70}
}


# ============================================================================
# MAPPING FUNCTIONS
# ============================================================================

def map_expert_labels_to_symbols(row: pd.Series) -> str:
    """
    Map expert labels (NDI/PDI) and sensor readings to semantic symbols.
    
    Implements the priority hierarchy from Section IV.A.1:
    1. Water deficit (highest urgency)
    2. pH imbalance
    3. Nutrient deficiency
    4. Heat stress
    5. Fungal risk
    
    Args:
        row: DataFrame row with sensor readings and expert labels
    
    Returns:
        str: Semantic symbol name
    """
    # Extract sensor values
    moisture = row['Moisture']
    ph = row['pH']
    nitrogen = row['N']
    temp = row['Temperature']
    humidity = row['Humidity']
    
    # Extract expert labels (if available)
    ndi_label = row.get('NDI_Label', 'Unknown')
    pdi_label = row.get('PDI_Label', 'Unknown')
    
    # ========================================================================
    # PRIORITY 1: Water Deficit + pH Imbalance
    # ========================================================================
    if moisture < THRESHOLDS['moisture']['low']:
        if ph < THRESHOLDS['ph']['acidic']:
            return 'water_deficit_acidic'
        elif ph > THRESHOLDS['ph']['alkaline']:
            return 'water_deficit_alkaline'
    
    # ========================================================================
    # PRIORITY 2: pH Imbalance (Adequate Moisture)
    # ========================================================================
    if moisture >= THRESHOLDS['moisture']['low']:
        if ph < THRESHOLDS['ph']['acidic']:
            return 'acidic_soil'
        elif ph > THRESHOLDS['ph']['alkaline']:
            return 'alkaline_soil'
    
    # ========================================================================
    # PRIORITY 3: Nutrient Deficiency (from expert label)
    # ========================================================================
    if ndi_label == 'High':
        return 'nutrient_deficiency'
    
    # ========================================================================
    # PRIORITY 4: Heat Stress
    # ========================================================================
    if temp > THRESHOLDS['temperature']['hot'] and humidity < THRESHOLDS['humidity']['dry']:
        return 'heat_stress'
    
    # ========================================================================
    # PRIORITY 5: Fungal Risk (PDI + Environmental Conditions)
    # Targets Botrytis cinerea: cool + humid + adequate moisture
    # ========================================================================
    if (pdi_label == 'High' and 
        humidity > THRESHOLDS['humidity']['humid'] and 
        temp < THRESHOLDS['temperature']['cool']):
        return 'fungal_risk'
    
    # ========================================================================
    # OPTIMAL CONDITION: All parameters within ideal ranges
    # ========================================================================
    if (THRESHOLDS['moisture']['low'] <= moisture <= THRESHOLDS['moisture']['high'] and
        THRESHOLDS['ph']['ideal_low'] <= ph <= THRESHOLDS['ph']['ideal_high'] and
        THRESHOLDS['nitrogen']['low'] <= nitrogen <= THRESHOLDS['nitrogen']['high'] and
        THRESHOLDS['temperature']['ideal_low'] <= temp <= THRESHOLDS['temperature']['ideal_high'] and
        THRESHOLDS['humidity']['ideal_low'] <= humidity <= THRESHOLDS['humidity']['ideal_high']):
        return 'optimal'
    
    # ========================================================================
    # AMBIGUOUS: Cannot determine clear semantic state
    # ========================================================================
    return 'ambiguous'


def calculate_statistics(df_original: pd.DataFrame, df_clean: pd.DataFrame) -> Dict:
    """
    Calculate preprocessing statistics for reporting.
    
    Args:
        df_original: Original dataset
        df_clean: Cleaned dataset after preprocessing
    
    Returns:
        dict: Statistics dictionary
    """
    total_original = len(df_original)
    total_clean = len(df_clean)
    removed = total_original - total_clean
    
    # Label distribution
    label_counts = df_clean['semantic_label'].value_counts()
    label_percentages = (label_counts / total_clean * 100).round(1)
    
    stats = {
        'original_samples': total_original,
        'cleaned_samples': total_clean,
        'removed_samples': removed,
        'removal_rate': round(removed / total_original * 100, 2),
        'label_distribution': dict(zip(label_counts.index, label_percentages.values))
    }
    
    return stats


def print_statistics(stats: Dict) -> None:
    """Print preprocessing statistics to console."""
    print("\n" + "="*70)
    print("DATASET PREPROCESSING STATISTICS")
    print("="*70)
    print(f"Original samples:        {stats['original_samples']:,}")
    print(f"After preprocessing:     {stats['cleaned_samples']:,}")
    print(f"Removed (ambiguous):     {stats['removed_samples']:,} ({stats['removal_rate']:.1f}%)")
    print("\n" + "-"*70)
    print("SEMANTIC LABEL DISTRIBUTION")
    print("-"*70)
    
    for label, percentage in sorted(stats['label_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
        count = int(stats['cleaned_samples'] * percentage / 100)
        bar = "█" * int(percentage / 2)
        print(f"{label:25s} {count:6,} ({percentage:5.1f}%) {bar}")
    
    print("="*70 + "\n")


def save_statistics(stats: Dict, filename: str) -> None:
    """Save statistics to text file."""
    with open(filename, 'w') as f:
        f.write("FuzSemCom Dataset Preprocessing Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Original samples:        {stats['original_samples']:,}\n")
        f.write(f"After preprocessing:     {stats['cleaned_samples']:,}\n")
        f.write(f"Removed (ambiguous):     {stats['removed_samples']:,} ({stats['removal_rate']:.1f}%)\n\n")
        f.write("Semantic Label Distribution:\n")
        f.write("-"*70 + "\n")
        
        for label, percentage in sorted(stats['label_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
            count = int(stats['cleaned_samples'] * percentage / 100)
            f.write(f"{label:25s} {count:6,} ({percentage:5.1f}%)\n")
        
        f.write("\nMapping Rules Applied:\n")
        f.write("-"*70 + "\n")
        f.write("1. Water deficit + pH imbalance (highest priority)\n")
        f.write("2. pH imbalance with adequate moisture\n")
        f.write("3. Nutrient deficiency (NDI_Label = High)\n")
        f.write("4. Heat stress (temp > 30°C, humidity < 60%)\n")
        f.write("5. Fungal risk (PDI_Label = High, humid, cool)\n")
        f.write("6. Optimal (all parameters in ideal ranges)\n")


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main() -> None:
    """
    Main preprocessing pipeline:
    1. Load raw dataset
    2. Remove missing values
    3. Map expert labels to semantic symbols
    4. Filter ambiguous samples
    5. Add symbol codes
    6. Save cleaned dataset and statistics
    """
    
    print("\n🌱 FuzSemCom Dataset Preprocessing Pipeline")
    print("="*70)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading dataset...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✓ Loaded {len(df):,} samples from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"✗ Error: File '{INPUT_FILE}' not found!")
        return
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # Step 2: Check required columns
    print("\n[2/6] Checking required columns...")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return
    print(f"✓ All required columns present: {REQUIRED_COLUMNS}")
    
    # Check for expert label columns (optional)
    has_expert_labels = all(col in df.columns for col in EXPERT_LABEL_COLUMNS)
    if not has_expert_labels:
        print(f"⚠ Warning: Expert labels {EXPERT_LABEL_COLUMNS} not found.")
        print("  Mapping will be based on sensor thresholds only.")
        # Add dummy columns
        df['NDI_Label'] = 'Unknown'
        df['PDI_Label'] = 'Unknown'
    
    # Step 3: Remove missing values
    print("\n[3/6] Removing missing values...")
    df_clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    removed_na = len(df) - len(df_clean)
    print(f"✓ Removed {removed_na:,} samples with missing values")
    
    # Step 4: Map to semantic symbols
    print("\n[4/6] Mapping expert labels to semantic symbols...")
    df_clean['semantic_label'] = df_clean.apply(map_expert_labels_to_symbols, axis=1)
    print(f"✓ Applied mapping rules (priority hierarchy)")
    
    # Step 5: Filter ambiguous samples
    print("\n[5/6] Filtering ambiguous samples...")
    df_final = df_clean[df_clean['semantic_label'] != 'ambiguous'].copy()
    removed_ambiguous = len(df_clean) - len(df_final)
    print(f"✓ Removed {removed_ambiguous:,} ambiguous samples ({removed_ambiguous/len(df_clean)*100:.1f}%)")
    
    # Step 6: Add symbol codes
    print("\n[6/6] Adding symbol codes...")
    df_final['symbol_code'] = df_final['semantic_label'].map(SYMBOL_MAP)
    print(f"✓ Mapped symbols to 1-byte hex codes")
    
    # Calculate statistics
    stats = calculate_statistics(df, df_final)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved {len(df_final):,} samples to {OUTPUT_FILE}")
    
    save_statistics(stats, STATS_FILE)
    print(f"✓ Saved statistics to {STATS_FILE}")
    
    # Print summary
    print_statistics(stats)
    
    # Validation checks
    print("🔍 Validation Checks:")
    print("-"*70)
    
    # Check expected distribution (from paper)
    expected = {
        'optimal': 24.1,
        'water_deficit_acidic': 17.9,
        'water_deficit_alkaline': 14.8,
        'nutrient_deficiency': 12.3,
        'acidic_soil': 8.7,
        'fungal_risk': 9.2,
        'alkaline_soil': 7.1,
        'heat_stress': 5.9
    }
    
    actual = stats['label_distribution']
    
    for label, exp_pct in expected.items():
        act_pct = actual.get(label, 0.0)
        diff = abs(act_pct - exp_pct)
        status = "✓" if diff < 5.0 else "⚠"
        print(f"{status} {label:25s} Expected: {exp_pct:5.1f}%  Actual: {act_pct:5.1f}%  Diff: {diff:+5.1f}%")
    
    print("\n✅ Preprocessing complete!\n")


if __name__ == "__main__":
    main()
