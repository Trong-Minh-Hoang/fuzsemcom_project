"""
01_data_exploration.py - Comprehensive Data Exploration & Analysis

Performs exploratory data analysis (EDA) on agriculture dataset:
- Dataset overview (shape, columns, types)
- Missing value analysis
- Statistical summary
- Distribution analysis
- Correlation analysis
- Label distribution (NDI/PDI)
- Data quality checks
- Visualization generation

Author: FuzSemCom Team
Date: 2025-11-16
"""

import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
DATA_PATH = ROOT / "data/raw/Agriculture_dataset_with_metadata.csv"
OUTPUT_DIR = ROOT / "results/eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Expected columns (from paper)
SENSOR_COLUMNS = ['Moisture', 'pH', 'N', 'Temperature', 'Humidity']
LABEL_COLUMNS = ['NDI_Label', 'PDI_Label', 'Semantic_Tag']
METADATA_COLUMNS = ['Zone_ID', 'Image_Source_ID', 'Image_Type', 
                   'NDVI', 'NDRE', 'RGB_Damage_Score', 'UAV_Timestamp']

# Valid ranges (from Table I in paper)
VALID_RANGES = {
    'Moisture': (0, 100),
    'pH': (4.0, 9.0),
    'N': (0, 300),
    'Temperature': (10, 40),
    'Humidity': (30, 100)
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> pd.DataFrame:
    """
    Load raw agriculture dataset.
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"✓ Loaded dataset from: {DATA_PATH}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


# ============================================================================
# BASIC OVERVIEW
# ============================================================================

def print_basic_info(df: pd.DataFrame) -> None:
    """
    Print basic dataset information.
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print(f"\nColumn Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    print(f"\nFirst 5 Rows:")
    print(df.head())


# ============================================================================
# MISSING VALUE ANALYSIS
# ============================================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in dataset.
    
    Args:
        df: Input dataframe
    
    Returns:
        pd.DataFrame: Missing value summary
    """
    print("\n" + "="*80)
    print("MISSING VALUE ANALYSIS")
    print("="*80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percent': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_df) == 0:
        print("✓ No missing values found!")
    else:
        print(f"⚠️  Found missing values in {len(missing_df)} columns:")
        print(missing_df.to_string(index=False))
        
        # Visualize
        if len(missing_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(missing_df['Column'], missing_df['Missing_Percent'])
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Values by Column')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved missing value plot to {FIGURES_DIR / 'missing_values.png'}")
    
    return missing_df


# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def print_statistical_summary(df: pd.DataFrame) -> None:
    """
    Print statistical summary of sensor columns.
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY (Sensor Columns)")
    print("="*80)
    
    sensor_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
    
    if not sensor_cols:
        print("⚠️  No sensor columns found")
        return
    
    summary = df[sensor_cols].describe()
    print(summary)
    
    # Check for outliers (values outside valid ranges)
    print(f"\n{'='*80}")
    print("RANGE VALIDATION")
    print("="*80)
    
    for col in sensor_cols:
        if col in VALID_RANGES:
            min_val, max_val = VALID_RANGES[col]
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                pct = out_of_range / len(df) * 100
                print(f"⚠️  {col}: {out_of_range:,} values ({pct:.2f}%) outside [{min_val}, {max_val}]")
            else:
                print(f"✓ {col}: All values within [{min_val}, {max_val}]")


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def plot_distributions(df: pd.DataFrame) -> None:
    """
    Plot distribution of sensor variables.
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)
    
    sensor_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
    
    if not sensor_cols:
        print("⚠️  No sensor columns found")
        return
    
    # Create subplots
    n_cols = len(sensor_cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
    
    for i, col in enumerate(sensor_cols):
        # Histogram
        axes[0, i].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, i].set_title(f'{col} Distribution')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(alpha=0.3)
        
        # Add valid range lines
        if col in VALID_RANGES:
            min_val, max_val = VALID_RANGES[col]
            axes[0, i].axvline(min_val, color='red', linestyle='--', alpha=0.5, label='Valid range')
            axes[0, i].axvline(max_val, color='red', linestyle='--', alpha=0.5)
            axes[0, i].legend()
        
        # Box plot
        axes[1, i].boxplot(df[col].dropna())
        axes[1, i].set_title(f'{col} Box Plot')
        axes[1, i].set_ylabel(col)
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution plots to {FIGURES_DIR / 'sensor_distributions.png'}")


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot correlation matrix of sensor variables.
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    sensor_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
    
    if len(sensor_cols) < 2:
        print("⚠️  Need at least 2 sensor columns for correlation")
        return
    
    # Calculate correlation
    corr = df[sensor_cols].corr()
    
    print("Correlation Matrix:")
    print(corr.round(3))
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix (Sensor Variables)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved correlation matrix to {FIGURES_DIR / 'correlation_matrix.png'}")
    
    # Find strong correlations
    print(f"\nStrong Correlations (|r| > 0.7):")
    strong_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                strong_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    
    if strong_corr:
        for col1, col2, r in strong_corr:
            print(f"  {col1} ↔ {col2}: r = {r:.3f}")
    else:
        print("  None found")


# ============================================================================
# LABEL DISTRIBUTION
# ============================================================================

def analyze_label_distribution(df: pd.DataFrame) -> None:
    """
    Analyze distribution of NDI/PDI labels and Semantic_Tag.
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*80)
    
    label_cols = [col for col in LABEL_COLUMNS if col in df.columns]
    
    if not label_cols:
        print("⚠️  No label columns found")
        return
    
    # Create subplots
    n_cols = len(label_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(label_cols):
        # Count distribution
        counts = df[col].value_counts()
        
        print(f"\n{col} Distribution:")
        for label, count in counts.items():
            pct = count / len(df) * 100
            print(f"  {label:30s}: {count:6,} ({pct:5.1f}%)")
        
        # Plot
        counts.plot(kind='bar', ax=axes[i], edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for j, (label, count) in enumerate(counts.items()):
            pct = count / len(df) * 100
            axes[i].text(j, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved label distribution to {FIGURES_DIR / 'label_distribution.png'}")


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

def perform_quality_checks(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: Input dataframe
    
    Returns:
        dict: Quality check results
    """
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    results = {}
    
    # 1. Duplicate rows
    duplicates = df.duplicated().sum()
    results['duplicates'] = duplicates
    if duplicates > 0:
        print(f"⚠️  Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    else:
        print(f"✓ No duplicate rows")
    
    # 2. Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    results['constant_columns'] = constant_cols
    if constant_cols:
        print(f"⚠️  Found {len(constant_cols)} constant columns: {constant_cols}")
    else:
        print(f"✓ No constant columns")
    
    # 3. High cardinality columns
    high_card_cols = [col for col in df.columns if df[col].nunique() > len(df) * 0.9]
    results['high_cardinality_columns'] = high_card_cols
    if high_card_cols:
        print(f"⚠️  Found {len(high_card_cols)} high-cardinality columns: {high_card_cols}")
    else:
        print(f"✓ No high-cardinality columns")
    
    # 4. Negative values in sensor columns
    sensor_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
    negative_cols = {}
    for col in sensor_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_cols[col] = neg_count
    
    results['negative_values'] = negative_cols
    if negative_cols:
        print(f"⚠️  Found negative values:")
        for col, count in negative_cols.items():
            print(f"    {col}: {count:,} negative values")
    else:
        print(f"✓ No negative values in sensor columns")
    
    # 5. Zero variance columns
    zero_var_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if df[col].std() == 0]
    results['zero_variance_columns'] = zero_var_cols
    if zero_var_cols:
        print(f"⚠️  Found {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
    else:
        print(f"✓ No zero-variance columns")
    
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_text_report(df: pd.DataFrame, quality_results: Dict) -> None:
    """
    Generate comprehensive text report.
    
    Args:
        df: Input dataframe
        quality_results: Quality check results
    """
    report_path = REPORTS_DIR / "eda_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Dataset overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"File: {DATA_PATH}\n")
        f.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # Columns
        f.write("2. COLUMNS\n")
        f.write("-"*80 + "\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:2d}. {col} ({df[col].dtype})\n")
        f.write("\n")
        
        # Missing values
        f.write("3. MISSING VALUES\n")
        f.write("-"*80 + "\n")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            f.write("No missing values\n\n")
        else:
            for col, count in missing[missing > 0].items():
                pct = count / len(df) * 100
                f.write(f"{col}: {count:,} ({pct:.2f}%)\n")
            f.write("\n")
        
        # Statistical summary
        f.write("4. STATISTICAL SUMMARY (Sensor Columns)\n")
        f.write("-"*80 + "\n")
        sensor_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
        if sensor_cols:
            f.write(df[sensor_cols].describe().to_string())
        f.write("\n\n")
        
        # Label distribution
        f.write("5. LABEL DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        label_cols = [col for col in LABEL_COLUMNS if col in df.columns]
        for col in label_cols:
            f.write(f"\n{col}:\n")
            counts = df[col].value_counts()
            for label, count in counts.items():
                pct = count / len(df) * 100
                f.write(f"  {label:30s}: {count:6,} ({pct:5.1f}%)\n")
        f.write("\n")
        
        # Quality checks
        f.write("6. DATA QUALITY\n")
        f.write("-"*80 + "\n")
        f.write(f"Duplicate rows: {quality_results['duplicates']:,}\n")
        f.write(f"Constant columns: {len(quality_results['constant_columns'])}\n")
        f.write(f"High-cardinality columns: {len(quality_results['high_cardinality_columns'])}\n")
        f.write(f"Columns with negative values: {len(quality_results['negative_values'])}\n")
        f.write(f"Zero-variance columns: {len(quality_results['zero_variance_columns'])}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n✓ Saved text report to {report_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """
    Main EDA pipeline.
    
    Steps:
    1. Load data
    2. Basic overview
    3. Missing value analysis
    4. Statistical summary
    5. Distribution analysis
    6. Correlation analysis
    7. Label distribution
    8. Quality checks
    9. Generate report
    """
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Basic overview
    print_basic_info(df)
    
    # Step 3: Missing values
    analyze_missing_values(df)
    
    # Step 4: Statistical summary
    print_statistical_summary(df)
    
    # Step 5: Distributions
    plot_distributions(df)
    
    # Step 6: Correlations
    plot_correlation_matrix(df)
    
    # Step 7: Label distribution
    analyze_label_distribution(df)
    
    # Step 8: Quality checks
    quality_results = perform_quality_checks(df)
    
    # Step 9: Generate report
    generate_text_report(df, quality_results)
    
    # Summary
    print("\n" + "="*80)
    print("✅ EDA COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  Figures:  {FIGURES_DIR}")
    print(f"  Reports:  {REPORTS_DIR}")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
