"""
06_ablation_study.py - Bayesian Optimization Ablation Study for FuzSemCom

Features:
- Optimizes 45 MF parameters using GP-based Bayesian Optimization
- Compares optimized vs baseline (paper) MFs
- Generates visualization of optimization progress
- Calculates improvement metrics
- Validates against expected results from paper (91.2% target)

Author: FuzSemCom Team
Date: 2025-11-16
"""

import sys
from pathlib import Path
import os
import json
import random
import time
import warnings

# Add the project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import skopt
_HAS_SKOPT = True
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.plots import plot_convergence, plot_objective
except ImportError:
    _HAS_SKOPT = False
    warnings.warn("scikit-optimize not found. Falling back to random search.")

# Import FIS builder
from src.fuzzy_engine import (
    create_fuzzy_system_with_params, 
    get_semantic_output,
    create_fuzzy_system  # Baseline system
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = ROOT / "data/processed/semantic_dataset.csv"
RESULTS_DIR = ROOT / "results" / "reports"
FIGURES_DIR = ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Expected results from paper Section IV.E
PAPER_BASELINE_ACCURACY = 0.887  # 88.7% (baseline MFs)
PAPER_OPTIMIZED_ACCURACY = 0.912  # 91.2% (after BO)
PAPER_IMPROVEMENT = 0.025  # 2.5 percentage points


# ============================================================================
# DATA LOADING
# ============================================================================

def load_validation_data(frac: float = 0.1, random_state: int = RANDOM_STATE):
    """
    Load validation subset for BO optimization.
    
    Args:
        frac: Fraction of dataset to use (default: 10%)
        random_state: Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Validation dataset
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    required_cols = ['soil_moisture', 'pH', 'nitrogen', 'temperature', 
                     'humidity', 'semantic_label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    
    # Sample validation subset
    df_val = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_val)} samples for validation from {DATA_PATH}")
    print(f"  Label distribution:")
    for label, count in df_val['semantic_label'].value_counts().items():
        print(f"    {label:30s}: {count:5d} ({count/len(df_val)*100:5.1f}%)")
    
    return df_val


# ============================================================================
# BASELINE EVALUATION
# ============================================================================

def evaluate_baseline_system(df_val):
    """
    Evaluate baseline FIS (paper MFs) on validation set.
    
    Returns:
        dict: Baseline metrics
    """
    print("\n" + "="*80)
    print("BASELINE SYSTEM EVALUATION (Paper MFs)")
    print("="*80)
    
    sim, out, antecedents, rules = create_fuzzy_system()
    
    preds = []
    true_labels = df_val['semantic_label'].tolist()
    
    for _, row in df_val.iterrows():
        inputs = {
            'soil_moisture': row['soil_moisture'],
            'pH': row['pH'],
            'nitrogen': row['nitrogen'],
            'temperature': row['temperature'],
            'humidity': row['humidity']
        }
        try:
            pred_label, confidence = get_semantic_output(sim, inputs, antecedents, rules)
            preds.append(pred_label)
        except Exception as e:
            preds.append("other")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds, 
                         labels=['optimal', 'water_deficit_acidic', 
                                'water_deficit_alkaline', 'acidic_soil',
                                'alkaline_soil', 'heat_stress', 
                                'nutrient_deficiency', 'fungal_risk'])
    
    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    
    baseline_results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': report
    }
    
    print(f"\n✓ Baseline Accuracy: {accuracy*100:.2f}%")
    print(f"  Expected (from paper): {PAPER_BASELINE_ACCURACY*100:.2f}%")
    print(f"  Difference: {abs(accuracy - PAPER_BASELINE_ACCURACY)*100:.2f}%")
    
    return baseline_results


# ============================================================================
# SEARCH SPACE DEFINITION
# ============================================================================

def build_search_space():
    """
    Build search space for 45 MF parameters.
    
    Order: [soil_moisture (9), pH (9), nitrogen (9), temperature (9), humidity (9)]
    Each variable: [dry/low/cool_l, _m, _r, ideal_l, _m, _r, wet/high/humid_l, _m, _r]
    
    Returns:
        list: 45 Real dimensions for skopt
    """
    space = []
    
    # Soil moisture (0-100%)
    space.extend([
        Real(0.0, 20.0, name='sm_dry_l'),
        Real(10.0, 35.0, name='sm_dry_m'),
        Real(20.0, 50.0, name='sm_dry_r'),
        Real(20.0, 50.0, name='sm_ideal_l'),
        Real(35.0, 70.0, name='sm_ideal_m'),
        Real(50.0, 80.0, name='sm_ideal_r'),
        Real(50.0, 80.0, name='sm_wet_l'),
        Real(65.0, 90.0, name='sm_wet_m'),
        Real(80.0, 100.0, name='sm_wet_r')
    ])
    
    # pH (4.0-9.0)
    space.extend([
        Real(4.0, 5.0, name='pH_acidic_l'),
        Real(4.5, 6.0, name='pH_acidic_m'),
        Real(5.0, 6.5, name='pH_acidic_r'),
        Real(5.5, 6.5, name='pH_ideal_l'),
        Real(6.0, 6.8, name='pH_ideal_m'),
        Real(6.5, 7.5, name='pH_ideal_r'),
        Real(6.5, 7.8, name='pH_alkaline_l'),
        Real(7.0, 8.0, name='pH_alkaline_m'),
        Real(7.5, 9.0, name='pH_alkaline_r')
    ])
    
    # Nitrogen (0-300 mg/kg)
    space.extend([
        Real(0.0, 30.0, name='N_low_l'),
        Real(10.0, 60.0, name='N_low_m'),
        Real(40.0, 80.0, name='N_low_r'),
        Real(30.0, 90.0, name='N_adequate_l'),
        Real(70.0, 110.0, name='N_adequate_m'),
        Real(90.0, 150.0, name='N_adequate_r'),
        Real(80.0, 200.0, name='N_high_l'),
        Real(150.0, 250.0, name='N_high_m'),
        Real(200.0, 300.0, name='N_high_r')
    ])
    
    # Temperature (10-40°C)
    space.extend([
        Real(10.0, 18.0, name='temp_cool_l'),
        Real(12.0, 24.0, name='temp_cool_m'),
        Real(20.0, 30.0, name='temp_cool_r'),
        Real(18.0, 30.0, name='temp_ideal_l'),
        Real(22.0, 30.0, name='temp_ideal_m'),
        Real(26.0, 35.0, name='temp_ideal_r'),
        Real(24.0, 38.0, name='temp_hot_l'),
        Real(32.0, 40.0, name='temp_hot_m'),
        Real(35.0, 40.0, name='temp_hot_r')
    ])
    
    # Humidity (30-100%)
    space.extend([
        Real(30.0, 50.0, name='humid_dry_l'),
        Real(35.0, 65.0, name='humid_dry_m'),
        Real(50.0, 70.0, name='humid_dry_r'),
        Real(45.0, 70.0, name='humid_ideal_l'),
        Real(60.0, 80.0, name='humid_ideal_m'),
        Real(65.0, 85.0, name='humid_ideal_r'),
        Real(60.0, 90.0, name='humid_humid_l'),
        Real(75.0, 95.0, name='humid_humid_m'),
        Real(85.0, 100.0, name='humid_humid_r')
    ])
    
    if len(space) != 45:
        raise RuntimeError(f"Expected 45 dimensions, got {len(space)}")
    
    print(f"✓ Search space defined: {len(space)} parameters")
    return space


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def evaluate_params_accuracy(params_vector, df_val, verbose=False):
    """
    Evaluate accuracy for given parameter vector.
    
    Args:
        params_vector: 45-element list/array of MF parameters
        df_val: Validation dataframe
        verbose: Print detailed errors
    
    Returns:
        float: Accuracy [0, 1]
    """
    # Enforce l <= m <= r for each MF
    params_list = list(params_vector)
    ordered_params = []
    for i in range(0, len(params_list), 3):
        l, m, r = params_list[i], params_list[i+1], params_list[i+2]
        x = sorted([float(l), float(m), float(r)])
        ordered_params.extend(x)
    
    try:
        sim, out, antecedents, rules = create_fuzzy_system_with_params(ordered_params)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to build FIS: {e}")
        return 0.0
    
    preds = []
    true_labels = df_val['semantic_label'].tolist()
    
    for _, row in df_val.iterrows():
        inputs = {
            'soil_moisture': row['soil_moisture'],
            'pH': row['pH'],
            'nitrogen': row['nitrogen'],
            'temperature': row['temperature'],
            'humidity': row['humidity']
        }
        try:
            pred_label, confidence = get_semantic_output(sim, inputs, antecedents, rules)
            preds.append(pred_label)
        except Exception:
            preds.append("other")
    
    if not preds:
        return 0.0
    
    accuracy = accuracy_score(true_labels, preds)
    return accuracy


# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

def run_bayesian_optimization(df_val, n_calls=50, n_initial_points=10, 
                              random_state=RANDOM_STATE):
    """
    Run GP-based Bayesian Optimization.
    
    Args:
        df_val: Validation dataset
        n_calls: Total number of evaluations
        n_initial_points: Random exploration before GP
        random_state: Random seed
    
    Returns:
        skopt.OptimizeResult: Optimization result
    """
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION (GP-Minimize)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Total evaluations:    {n_calls}")
    print(f"  Initial random:       {n_initial_points}")
    print(f"  GP iterations:        {n_calls - n_initial_points}")
    print(f"  Validation samples:   {len(df_val)}")
    print("")
    
    space = build_search_space()
    
    # Track progress
    iteration = [0]
    best_acc = [0.0]
    
    def objective(x):
        iteration[0] += 1
        acc = evaluate_params_accuracy(x, df_val)
        
        if acc > best_acc[0]:
            best_acc[0] = acc
            status = "✓ NEW BEST"
        else:
            status = ""
        
        print(f"[{iteration[0]:3d}/{n_calls}] Accuracy: {acc*100:6.2f}% | "
              f"Best: {best_acc[0]*100:6.2f}% {status}")
        
        return -acc  # Minimize negative accuracy
    
    print("Starting optimization...")
    start_time = time.time()
    
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=False,  # We handle our own logging
        n_jobs=1  # Single-threaded for reproducibility
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Optimization complete in {elapsed:.1f}s")
    print(f"  Best accuracy: {-result.fun*100:.2f}%")
    print(f"  Baseline:      {PAPER_BASELINE_ACCURACY*100:.2f}%")
    print(f"  Improvement:   {(-result.fun - PAPER_BASELINE_ACCURACY)*100:+.2f}%")
    
    return result


# ============================================================================
# FALLBACK RANDOM SEARCH
# ============================================================================

def run_random_search(df_val, n_calls=50, random_state=RANDOM_STATE):
    """
    Fallback random search if skopt unavailable.
    """
    print("\n" + "="*80)
    print("RANDOM SEARCH (Fallback)")
    print("="*80)
    
    rng = random.Random(random_state)
    np.random.seed(random_state)
    
    best_acc = 0.0
    best_params = None
    history = []
    
    # Define bounds (matching build_search_space order)
    bounds = (
        # Soil moisture
        [(0.0, 20.0), (10.0, 35.0), (20.0, 50.0),
         (20.0, 50.0), (35.0, 70.0), (50.0, 80.0),
         (50.0, 80.0), (65.0, 90.0), (80.0, 100.0)] +
        # pH
        [(4.0, 5.0), (4.5, 6.0), (5.0, 6.5),
         (5.5, 6.5), (6.0, 6.8), (6.5, 7.5),
         (6.5, 7.8), (7.0, 8.0), (7.5, 9.0)] +
        # Nitrogen
        [(0.0, 30.0), (10.0, 60.0), (40.0, 80.0),
         (30.0, 90.0), (70.0, 110.0), (90.0, 150.0),
         (80.0, 200.0), (150.0, 250.0), (200.0, 300.0)] +
        # Temperature
        [(10.0, 18.0), (12.0, 24.0), (20.0, 30.0),
         (18.0, 30.0), (22.0, 30.0), (26.0, 35.0),
         (24.0, 38.0), (32.0, 40.0), (35.0, 40.0)] +
        # Humidity
        [(30.0, 50.0), (35.0, 65.0), (50.0, 70.0),
         (45.0, 70.0), (60.0, 80.0), (65.0, 85.0),
         (60.0, 90.0), (75.0, 95.0), (85.0, 100.0)]
    )
    
    for it in range(n_calls):
        vec = [rng.uniform(lo, hi) for (lo, hi) in bounds]
        acc = evaluate_params_accuracy(vec, df_val)
        
        # Enforce ordering
        ordered_vec = []
        for i in range(0, len(vec), 3):
            l, m, r = vec[i], vec[i+1], vec[i+2]
            x = sorted([float(l), float(m), float(r)])
            ordered_vec.extend(x)
        
        history.append({
            "params": [round(float(v), 6) for v in ordered_vec],
            "accuracy": round(float(acc), 6)
        })
        
        if acc > best_acc:
            best_acc = acc
            best_params = ordered_vec
            status = "✓ NEW BEST"
        else:
            status = ""
        
        print(f"[{it+1:3d}/{n_calls}] Accuracy: {acc*100:6.2f}% | "
              f"Best: {best_acc*100:6.2f}% {status}")
    
    return {
        "best_params": best_params,
        "best_acc": best_acc,
        "history": history
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_optimization_results(result, baseline_acc, output_path):
    """
    Generate visualization of optimization progress.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Convergence plot
    ax1 = axes[0, 0]
    iterations = range(1, len(result.func_vals) + 1)
    accuracies = [-val for val in result.func_vals]
    best_so_far = np.maximum.accumulate(accuracies)
    
    ax1.plot(iterations, accuracies, 'o-', alpha=0.6, label='Trial accuracy')
    ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
    ax1.axhline(baseline_acc, color='green', linestyle='--', 
                label=f'Baseline ({baseline_acc*100:.1f}%)')
    ax1.axhline(PAPER_OPTIMIZED_ACCURACY, color='orange', linestyle='--',
                label=f'Paper target ({PAPER_OPTIMIZED_ACCURACY*100:.1f}%)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement distribution
    ax2 = axes[0, 1]
    improvements = [(acc - baseline_acc) * 100 for acc in accuracies]
    ax2.hist(improvements, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', label='Baseline')
    ax2.axvline((PAPER_OPTIMIZED_ACCURACY - baseline_acc) * 100, 
                color='orange', linestyle='--', label='Paper improvement')
    ax2.set_xlabel('Improvement over Baseline (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Improvements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs iteration (scatter)
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(accuracies)))
    ax3.scatter(iterations, accuracies, c=colors, s=50, alpha=0.6)
    ax3.plot(iterations, best_so_far, 'r-', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Exploration Pattern')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    OPTIMIZATION SUMMARY
    {'='*40}
    
    Baseline Accuracy:    {baseline_acc*100:.2f}%
    Best Accuracy:        {max(accuracies)*100:.2f}%
    Improvement:          {(max(accuracies) - baseline_acc)*100:+.2f}%
    
    Paper Target:         {PAPER_OPTIMIZED_ACCURACY*100:.2f}%
    Target Achieved:      {'✓ Yes' if max(accuracies) >= PAPER_OPTIMIZED_ACCURACY else '✗ No'}
    
    Total Evaluations:    {len(accuracies)}
    Mean Accuracy:        {np.mean(accuracies)*100:.2f}%
    Std Accuracy:         {np.std(accuracies)*100:.2f}%
    
    Best Iteration:       {np.argmax(accuracies) + 1}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved optimization plot to {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(n_calls=50, n_initial_points=10):
    """
    Main ablation study pipeline.
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Bayesian Optimization of Membership Functions")
    print("="*80)
    print(f"Target: Improve from {PAPER_BASELINE_ACCURACY*100:.1f}% to "
          f"{PAPER_OPTIMIZED_ACCURACY*100:.1f}% (+{PAPER_IMPROVEMENT*100:.1f}%)")
    print("")
    
    start_time = time.time()
    
    # Step 1: Load data
    print("[1/5] Loading validation data...")
    df_val = load_validation_data(frac=0.1, random_state=RANDOM_STATE)
    
    # Step 2: Evaluate baseline
    print("\n[2/5] Evaluating baseline system...")
    baseline_results = evaluate_baseline_system(df_val)
    baseline_acc = baseline_results['accuracy']
    
    # Step 3: Run optimization
    print("\n[3/5] Running optimization...")
    
    if _HAS_SKOPT:
        result = run_bayesian_optimization(df_val, n_calls, n_initial_points, RANDOM_STATE)
        
        best_acc = -result.fun
        best_params = result.x
        
        # Prepare history
        history = []
        for vec, funval in zip(result.x_iters, result.func_vals):
            ordered_vec = []
            for i in range(0, len(vec), 3):
                l, m, r = vec[i], vec[i+1], vec[i+2]
                x = sorted([float(l), float(m), float(r)])
                ordered_vec.extend(x)
            history.append({
                "params": [round(float(v), 6) for v in ordered_vec],
                "accuracy": round(float(-funval), 6)
            })
        
        output = {
            "method": "gp_minimize",
            "baseline_accuracy": round(float(baseline_acc), 6),
            "best_accuracy": round(float(best_acc), 6),
            "improvement": round(float(best_acc - baseline_acc), 6),
            "improvement_percent": round(float((best_acc - baseline_acc) * 100), 2),
            "best_params": [round(float(x), 6) for x in best_params],
            "optimization_history": history,
            "paper_target_achieved": best_acc >= PAPER_OPTIMIZED_ACCURACY
        }
        
        # Step 4: Visualize
        print("\n[4/5] Generating visualizations...")
        plot_path = FIGURES_DIR / "ablation_optimization_results.png"
        plot_optimization_results(result, baseline_acc, plot_path)
        
    else:
        result = run_random_search(df_val, n_calls, RANDOM_STATE)
        
        output = {
            "method": "random_search",
            "baseline_accuracy": round(float(baseline_acc), 6),
            "best_accuracy": round(float(result["best_acc"]), 6),
            "improvement": round(float(result["best_acc"] - baseline_acc), 6),
            "improvement_percent": round(float((result["best_acc"] - baseline_acc) * 100), 2),
            "best_params": [round(float(x), 6) for x in result["best_params"]],
            "optimization_history": result["history"],
            "paper_target_achieved": result["best_acc"] >= PAPER_OPTIMIZED_ACCURACY
        }
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    out_path = RESULTS_DIR / "bo_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"✓ Results saved to {out_path}")
    
    # Print final summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Baseline Accuracy:        {baseline_acc*100:.2f}%")
    print(f"Optimized Accuracy:       {output['best_accuracy']*100:.2f}%")
    print(f"Improvement:              {output['improvement_percent']:+.2f}%")
    print(f"Paper Target:             {PAPER_OPTIMIZED_ACCURACY*100:.2f}%")
    print(f"Target Achieved:          {'✓ Yes' if output['paper_target_achieved'] else '✗ No'}")
    print(f"Total Time:               {elapsed:.1f}s")
    print(f"Evaluations:              {n_calls}")
    print("="*80 + "\n")
    
    return output


if __name__ == "__main__":
    # Run with default parameters
    # Increase n_calls for better results (e.g., 100-200)
    main(n_calls=50, n_initial_points=10)
