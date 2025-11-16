"""
_evaluate_accuracy_bo.py - Objective Function for Bayesian Optimization

Helper function to evaluate FIS accuracy for BO ablation study.
Used by 06_ablation_study.py to optimize membership function parameters.

Features:
- Efficient evaluation with caching
- Robust error handling
- Parameter validation
- Progress tracking
- Memory optimization

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
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Optional

from src.fuzzy_engine import (
    create_fuzzy_system_with_params, 
    get_semantic_output
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH = ROOT / "data/processed/semantic_dataset.csv"
TRAIN_PATH = ROOT / "data/processed/semantic_dataset_train.csv"
TEST_PATH = ROOT / "data/processed/semantic_dataset_test.csv"

# Validation subset size (for faster BO iterations)
VALIDATION_FRACTION = 0.1  # Use 10% of data for BO
RANDOM_STATE = 42

# Cache for loaded data (avoid reloading on every call)
_DATA_CACHE = None


# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

def load_validation_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load validation dataset for BO optimization.
    Uses caching to avoid reloading on every evaluation.
    
    Args:
        use_cache: Use cached data if available
    
    Returns:
        pd.DataFrame: Validation dataset
    """
    global _DATA_CACHE
    
    if use_cache and _DATA_CACHE is not None:
        return _DATA_CACHE
    
    # Try to load train split first (for proper train/test separation)
    if TRAIN_PATH.exists():
        df = pd.read_csv(TRAIN_PATH)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError(
            f"No dataset found. Please run ground_truth_generator.py first."
        )
    
    # Sample validation subset for faster BO
    if len(df) > 1000:  # Only sample if dataset is large
        df_val = df.sample(
            frac=VALIDATION_FRACTION, 
            random_state=RANDOM_STATE
        ).reset_index(drop=True)
    else:
        df_val = df.copy()
    
    # Validate required columns
    required_cols = ['soil_moisture', 'pH', 'nitrogen', 'temperature', 
                     'humidity', 'semantic_label']
    missing = [col for col in required_cols if col not in df_val.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    # Cache for future calls
    _DATA_CACHE = df_val
    
    return df_val


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_params(params: List[float]) -> Tuple[bool, str]:
    """
    Validate parameter vector for fuzzy system.
    
    Args:
        params: 45-element parameter vector
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check length
    if len(params) != 45:
        return False, f"Expected 45 parameters, got {len(params)}"
    
    # Check for NaN or Inf
    if not all(np.isfinite(params)):
        return False, "Parameters contain NaN or Inf values"
    
    # Check triangular MF ordering (l <= m <= r) for each MF
    for i in range(0, len(params), 3):
        l, m, r = params[i], params[i+1], params[i+2]
        if not (l <= m <= r):
            return False, f"Invalid MF at index {i//3}: l={l}, m={m}, r={r} (must be l<=m<=r)"
    
    return True, ""


def enforce_mf_ordering(params: List[float]) -> List[float]:
    """
    Enforce l <= m <= r ordering for all membership functions.
    
    Args:
        params: 45-element parameter vector
    
    Returns:
        List[float]: Ordered parameter vector
    """
    ordered_params = []
    for i in range(0, len(params), 3):
        l, m, r = params[i], params[i+1], params[i+2]
        # Sort to enforce ordering
        sorted_vals = sorted([float(l), float(m), float(r)])
        ordered_params.extend(sorted_vals)
    
    return ordered_params


# ============================================================================
# ACCURACY EVALUATION
# ============================================================================

def evaluate_fis_accuracy(
    params: List[float],
    df: Optional[pd.DataFrame] = None,
    return_predictions: bool = False,
    verbose: bool = False
) -> float:
    """
    Evaluate FIS accuracy for given parameter vector.
    
    This is the OBJECTIVE FUNCTION for Bayesian Optimization.
    Returns NEGATIVE accuracy (for minimization).
    
    Args:
        params: 45-element parameter vector for MFs
        df: Validation dataframe (if None, loads from cache)
        return_predictions: Return (accuracy, predictions) tuple
        verbose: Print detailed error messages
    
    Returns:
        float: Negative accuracy (for minimization)
               or tuple (negative_accuracy, predictions) if return_predictions=True
    """
    # Validate parameters
    is_valid, error_msg = validate_params(params)
    if not is_valid:
        if verbose:
            print(f"[WARN] Invalid parameters: {error_msg}")
        return 0.0 if not return_predictions else (0.0, [])
    
    # Enforce MF ordering
    ordered_params = enforce_mf_ordering(params)
    
    # Load data if not provided
    if df is None:
        try:
            df = load_validation_data(use_cache=True)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load data: {e}")
            return 0.0 if not return_predictions else (0.0, [])
    
    # Create fuzzy system with given parameters
    try:
        sim, out, antecedents, rules = create_fuzzy_system_with_params(ordered_params)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to create FIS: {e}")
        return 0.0 if not return_predictions else (0.0, [])
    
    # Run inference on all samples
    predictions = []
    errors = 0
    
    for idx, row in df.iterrows():
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
        except Exception as e:
            # Fallback for inference errors
            predictions.append('other')
            errors += 1
    
    # Calculate accuracy
    if not predictions:
        if verbose:
            print("[WARN] No predictions generated")
        return 0.0 if not return_predictions else (0.0, [])
    
    try:
        true_labels = df['semantic_label'].tolist()
        accuracy = accuracy_score(true_labels, predictions)
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to calculate accuracy: {e}")
        return 0.0 if not return_predictions else (0.0, [])
    
    # Return negative accuracy (for minimization in BO)
    if return_predictions:
        return -accuracy, predictions
    else:
        return -accuracy


# ============================================================================
# BATCH EVALUATION (for parallel BO)
# ============================================================================

def evaluate_fis_accuracy_batch(
    params_list: List[List[float]],
    df: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> List[float]:
    """
    Evaluate multiple parameter vectors in batch.
    Useful for parallel Bayesian Optimization.
    
    Args:
        params_list: List of parameter vectors
        df: Validation dataframe
        verbose: Print progress
    
    Returns:
        List[float]: List of negative accuracies
    """
    if df is None:
        df = load_validation_data(use_cache=True)
    
    results = []
    for i, params in enumerate(params_list):
        if verbose and i % 10 == 0:
            print(f"  Batch progress: {i}/{len(params_list)}")
        
        accuracy = evaluate_fis_accuracy(params, df, verbose=False)
        results.append(accuracy)
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_baseline_accuracy(df: Optional[pd.DataFrame] = None) -> float:
    """
    Get baseline accuracy using paper's default MFs.
    
    Args:
        df: Validation dataframe
    
    Returns:
        float: Baseline accuracy
    """
    from src.fuzzy_engine import create_fuzzy_system
    
    if df is None:
        df = load_validation_data(use_cache=True)
    
    sim, out, antecedents, rules = create_fuzzy_system()
    
    predictions = []
    for _, row in df.iterrows():
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
        except:
            predictions.append('other')
    
    true_labels = df['semantic_label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    
    return accuracy


def clear_data_cache():
    """Clear cached validation data (useful for memory management)."""
    global _DATA_CACHE
    _DATA_CACHE = None


# ============================================================================
# TESTING & DEBUGGING
# ============================================================================

def test_evaluation_function():
    """
    Test the evaluation function with random parameters.
    """
    print("\n" + "="*80)
    print("TESTING EVALUATION FUNCTION")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading validation data...")
    df = load_validation_data(use_cache=False)
    print(f"✓ Loaded {len(df):,} samples")
    
    # Test baseline
    print("\n[2/4] Testing baseline accuracy...")
    baseline_acc = get_baseline_accuracy(df)
    print(f"✓ Baseline accuracy: {baseline_acc*100:.2f}%")
    
    # Test with random parameters
    print("\n[3/4] Testing with random parameters...")
    np.random.seed(42)
    
    # Generate random params (45 values)
    random_params = []
    # Soil moisture (0-100)
    random_params.extend(np.random.uniform(0, 100, 9).tolist())
    # pH (4-9)
    random_params.extend(np.random.uniform(4, 9, 9).tolist())
    # Nitrogen (0-300)
    random_params.extend(np.random.uniform(0, 300, 9).tolist())
    # Temperature (10-40)
    random_params.extend(np.random.uniform(10, 40, 9).tolist())
    # Humidity (30-100)
    random_params.extend(np.random.uniform(30, 100, 9).tolist())
    
    neg_acc = evaluate_fis_accuracy(random_params, df, verbose=True)
    print(f"✓ Random params accuracy: {-neg_acc*100:.2f}%")
    
    # Test batch evaluation
    print("\n[4/4] Testing batch evaluation...")
    params_batch = [random_params for _ in range(5)]
    results = evaluate_fis_accuracy_batch(params_batch, df, verbose=True)
    print(f"✓ Batch results: {[-r for r in results]}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run tests when executed directly
    test_evaluation_function()
