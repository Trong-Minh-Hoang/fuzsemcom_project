fuzsemcom_project/
├── README.md                           # Project overview & quick start
├── requirements.txt                    # Dependencies: pandas, scikit-fuzzy, numpy, matplotlib
│
├── src/                                # Core implementation (không chạy trực tiếp)
│   ├── __init__.py
│   ├── fuzzy_engine.py                # Đổi tên từ fuzzy_system_optimized.py
│   └── ground_truth_generator.py      # Logic tạo labels (từ 04_ground_truth.py)
│
├── scripts/                            # Scripts thực thi theo pipeline
│   ├── 01_data_exploration.py         # Explore dataset
│   ├── 02_data_preprocessing.py       # Clean & filter data
│   ├── 03_generate_ground_truth.py    # Generate semantic labels
│   ├── 04_evaluate_fse.py             # Main evaluation (đổi từ 05_evaluate_fse.py)
│   ├── 05_compare_with_deepsc.py      # Comparison (đổi từ 06_deepsc_comparison.py)
│   ├── 06_ablation_study.py           # Optional: ablation analysis
│   └── debug_prediction.py            # Debug tool (từ debug_optimized.py)
│
├── data/                               # Data directory (gitignored nếu dataset lớn)
│   ├── raw/
│   │   └── Agriculture_dataset_with_metadata.csv  # Dataset gốc (download từ IEEE DataPort)
│   └── processed/
│       └── semantic_dataset.csv       # Output sau khi chạy script 03
│
├── results/                            # Results & outputs (gitignored)
│   ├── figures/                       # All plots & visualizations
│   │   ├── fse_confusion_matrix.png
│   │   ├── deepsc_confusion_matrix.png
│   │   └── comparison_overview.png
│   └── reports/
│       ├── fse_evaluation_results.json
│       ├── deepsc_comparison_results.json
│       └── experiment_report.docx     # Auto-generated report
│
├── docs/
│   ├── ICC_ENGLAND_2026.pdf           # Bài báo gốc
│   └── student_guide_2026.pdf         # Hướng dẫn (đổi tên từ guide_2026.pdf)
│
└── .gitignore                         # Bỏ qua data/raw, results/, *.pyc


pandas>=2.0.0
numpy>=1.24.0
scikit-fuzzy>=0.4.2
scikit-learn>=1.3.0
tensorflow>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-optimize>=0.9.0

Step 1: Data Exploration (EDA)
   ↓
Step 2: Data Preprocessing  
   ↓
Step 3: Ground Truth Generation (Fuzzy Inference)
   ↓
Step 4: Evaluate FSE (Fuzzy Semantic Encoder)
   ↓
Step 5: Train Neural Decoder
   ↓
Step 6: Ablation Study (Bayesian Optimization)
   ↓
Step 7: Generate Final Report & Figures

STEP 1: DATA EXPLORATION
Mục đích:
Hiểu cấu trúc dataset
Kiểm tra missing values
Phân tích distribution
Validate ranges
Chạy:

cd scripts
python 01_data_exploration.py

✅ CHECKPOINTS:
Console Output:
================================================================================
EXPLORATORY DATA ANALYSIS PIPELINE
================================================================================

LOADING DATASET
✓ Loaded dataset from: data/raw/Agriculture_dataset_with_metadata.csv
  Shape: 60,000 rows × 24 columns
  Memory usage: 11.02 MB

DATASET OVERVIEW
Shape: 60,000 rows × 24 columns
Column Names (24): Zone_ID, Image_Source_ID, ... [danh sách đầy đủ]

MISSING VALUE ANALYSIS
⚠️  Found missing values in 5 columns:
  Migration_Timestamp: 57,234 (95.39%)
  NDRE: 30,000 (50.00%)
  ...

STATISTICAL SUMMARY
          Moisture          pH           N  Temperature    Humidity
count  60000.000000  60000.000  60000.000  60000.000000  60000.000
mean      22.456789    6.523     48.234      28.567890    62.345
...

RANGE VALIDATION
✓ Moisture: All values within [0, 100]
✓ pH: All values within [4.0, 9.0]
✓ N: All values within [0, 300]
✓ Temperature: All values within [10, 40]
✓ Humidity: All values within [30, 100]

LABEL DISTRIBUTION
NDI_Label Distribution:
  High  : 18,234 (30.4%)
  Medium: 24,567 (41.0%)
  Low   : 17,199 (28.7%)

✅ EDA COMPLETE
Output directory: results/eda
Files Created:
Sao chép
results/eda/
├── figures/
│   ├── missing_values.png          # ✅ Bar chart
│   ├── sensor_distributions.png    # ✅ 5 histograms + boxplots
│   ├── correlation_matrix.png      # ✅ Heatmap
│   └── label_distribution.png      # ✅ NDI/PDI/Semantic bars
└── reports/
    └── eda_report.txt              # ✅ Text summary
Kiểm tra:
Sao chép
# Check files exist
ls -lh results/eda/figures/
ls -lh results/eda/reports/

# View report
cat results/eda/reports/eda_report.txt
Expected Results (so với paper):
Metric	Expected	Your Result	✅/❌
Total samples	~60,000	?	
Moisture range	[0, 100]	?	
pH range	[4.0, 9.0]	?	
N range	[0, 300]	?	
Temp range	[10, 40]	?	
Humidity range	[30, 100]	?	

STEP 2: DATA PREPROCESSING
Mục đích:
Map NDI/PDI labels → semantic symbols
Apply priority hierarchy (Table III)
Validate fuzzy inference
Chạy:
Sao chép
python 02_data_preprocessing.py
✅ CHECKPOINTS:
Console Output:
Sao chép
================================================================================
DATA PREPROCESSING PIPELINE
================================================================================

[1/5] Loading raw data...
✓ Loaded 60,000 samples

[2/5] Validating data...
✓ Removed 1,579 rows with missing values
✓ Removed 312 rows with out-of-range values
✓ Final dataset: 58,109 valid samples

[3/5] Mapping NDI/PDI labels to semantic symbols...
✓ Applied priority hierarchy (Table III)
✓ Label distribution:
    optimal                    : 14,080 (24.2%)
    water_deficit_acidic       : 10,457 (18.0%)
    water_deficit_alkaline     :  8,645 (14.9%)
    nutrient_deficiency        :  7,186 (12.4%)
    fungal_risk                :  5,375 ( 9.2%)
    acidic_soil                :  5,083 ( 8.7%)
    alkaline_soil              :  4,148 ( 7.1%)
    heat_stress                :  3,447 ( 5.9%)
    other                      :  3,688 ( 6.3%)

[4/5] Validating with fuzzy inference...
✓ Fuzzy agreement: 88.7%

[5/5] Saving preprocessed data...
✓ Saved to data/processed/semantic_dataset_preprocessed.csv

✅ PREPROCESSING COMPLETE
Files Created:
Sao chép
data/processed/
├── semantic_dataset_preprocessed.csv   # ✅ Main output
└── preprocessing_stats.txt             # ✅ Statistics
Kiểm tra:
Sao chép
# Check file
head -20 data/processed/semantic_dataset_preprocessed.csv

# Check stats
cat data/processed/preprocessing_stats.txt

# Verify label distribution
python -c "
import pandas as pd
df = pd.read_csv('data/processed/semantic_dataset_preprocessed.csv')
print(df['semantic_label'].value_counts())
"
Expected Results (so với paper Table IV):
Label	Expected %	Your %	✅/❌
optimal	24.1%	?	
water_deficit_acidic	17.9%	?	
water_deficit_alkaline	14.8%	?	
nutrient_deficiency	12.3%	?	
fungal_risk	9.2%	?	
acidic_soil	8.7%	?	
alkaline_soil	7.1%	?	
heat_stress	5.9%	?	
🔥 STEP 3: GROUND TRUTH GENERATION
Mục đích:
Generate semantic labels using fuzzy inference
Split train/test (80/20)
Calculate confidence scores
Chạy:
Sao chép
python ground_truth_generator.py
✅ CHECKPOINTS:
Console Output:
Sao chép
================================================================================
FUZSEMCOM GROUND TRUTH GENERATION PIPELINE
================================================================================

[1/6] Loading raw data...
✓ Loaded 60,000 samples

[2/6] Validating data...
✓ Final dataset: 58,421 valid samples

[3/6] Generating semantic labels...
🔮 Generating semantic labels using fuzzy inference...
✓ Fuzzy system initialized (expert-defined membership functions)
✓ Generated labels for 58,421 samples

[4/6] Saving full labeled dataset...
✓ Saved to data/processed/semantic_dataset_fuzzy.csv

[5/6] Splitting train/test (80/20)...
✓ Train: 46,736 samples → semantic_dataset_train.csv
✓ Test:  11,685 samples → semantic_dataset_test.csv

[6/6] Generating statistics...
LABEL DISTRIBUTION
optimal                        14,080 (24.1%) ████████████
water_deficit_acidic           10,457 (17.9%) ████████
...

CONFIDENCE STATISTICS
Mean Confidence:    187.3/255 (73.5%)
Median Confidence:  201.0/255
Min Confidence:     45/255
Max Confidence:     255/255

✅ GROUND TRUTH GENERATION COMPLETE
Files Created:
Sao chép
data/processed/
├── semantic_dataset_fuzzy.csv          # ✅ Full dataset
├── semantic_dataset_train.csv          # ✅ Training split
├── semantic_dataset_test.csv           # ✅ Test split
└── fuzzy_generation_stats.txt          # ✅ Statistics
Kiểm tra:
Sao chép
# Check file sizes
wc -l data/processed/semantic_dataset_*.csv

# Expected:
# 58,422 semantic_dataset_fuzzy.csv (header + 58,421 rows)
# 46,737 semantic_dataset_train.csv
# 11,686 semantic_dataset_test.csv

# Check columns
head -1 data/processed/semantic_dataset_fuzzy.csv

# Expected columns:
# soil_moisture,pH,nitrogen,temperature,humidity,semantic_label,confidence

# Verify confidence distribution
python -c "
import pandas as pd
df = pd.read_csv('data/processed/semantic_dataset_fuzzy.csv')
print('Mean confidence:', df['confidence'].mean())
print('Median confidence:', df['confidence'].median())
"
Expected Results:
Metric	Expected	Your Result	✅/❌
Total samples	~58,000	?	
Train samples	~46,000	?	
Test samples	~11,000	?	
Mean confidence	180-190/255	?	
Label: optimal	~24%	?	
🔥 STEP 4: EVALUATE FSE (Fuzzy Semantic Encoder)
Mục đích:
Evaluate fuzzy inference accuracy
Generate confusion matrix
Analyze confidence scores
Compare with paper (88.7%)
Chạy:
Sao chép
python 04_evaluate_fse.py
✅ CHECKPOINTS:
Console Output:
Sao chép
================================================================================
FUZZY SEMANTIC ENCODER EVALUATION
================================================================================

LOADING TEST DATA
✓ Loaded test dataset: data/processed/semantic_dataset_test.csv
  Samples: 11,685

RUNNING FUZZY INFERENCE
✓ Fuzzy system initialized
Processing 11,685 samples...
  Progress: 1,000/11,685 (8.6%)
  ...
✓ Inference complete

CALCULATING METRICS
✓ Overall Accuracy: 88.73%
  Expected (paper): 88.70%
  Difference:       +0.03%

✓ Confidence Statistics:
  Mean:   187.3/255 (73.5%)
  Median: 201.0/255
  Std:    42.1
  Range:  [45, 255]

GENERATING VISUALIZATIONS
✓ Saved confusion matrix to results/figures/fse_confusion_matrix.png
✓ Saved per-class metrics to results/figures/fse_per_class_metrics.png
✓ Saved confidence distribution to results/figures/fse_confidence_distribution.png
✓ Saved symbol distribution to results/figures/fse_symbol_distribution.png

SAVING RESULTS
✓ Saved JSON results to results/reports/fse_evaluation_results.json
✓ Saved text report to results/reports/fse_evaluation_report.txt

================================================================================
EVALUATION COMPLETE
================================================================================
Overall Accuracy:     88.73%
Expected (paper):     88.70%
Difference:           +0.03%
Mean Confidence:      187.3/255
Inference Errors:     0
Files Created:
Sao chép
results/
├── figures/
│   ├── fse_confusion_matrix.png          # ✅ Normalized heatmap
│   ├── fse_per_class_metrics.png         # ✅ Precision/Recall/F1 bars
│   ├── fse_confidence_distribution.png   # ✅ Histogram + boxplot
│   └── fse_symbol_distribution.png       # ✅ Symbol frequency
└── reports/
    ├── fse_evaluation_results.json       # ✅ JSON metrics
    └── fse_evaluation_report.txt         # ✅ Text report
Kiểm tra:
Sao chép
# View JSON results
cat results/reports/fse_evaluation_results.json | python -m json.tool

# View text report
cat results/reports/fse_evaluation_report.txt

# Check accuracy
python -c "
import json
with open('results/reports/fse_evaluation_results.json') as f:
    data = json.load(f)
    print(f\"Accuracy: {data['accuracy']*100:.2f}%\")
    print(f\"Expected: {data['paper_comparison']['expected_accuracy']*100:.2f}%\")
"
Expected Results (so với paper Section IV.D):
Metric	Expected	Your Result	✅/❌
Overall Accuracy	88.7%	?	
Optimal Precision	~92%	?	
Optimal Recall	~91%	?	
Mean Confidence	180-190/255	?	
Inference Errors	0	?	
Visual Checks:
Sao chép
# Open figures
open results/figures/fse_confusion_matrix.png
open results/figures/fse_per_class_metrics.png
Confusion matrix should show:

Diagonal values > 0.85 (high accuracy)
Off-diagonal values < 0.10 (low confusion)
Optimal class: highest accuracy (~92%)
🔥 STEP 5: TRAIN NEURAL DECODER
Mục đích:
Train LSTM decoder (symbol → sensor values)
Evaluate reconstruction accuracy
Compare with paper (RMSE, MAE)
Chạy:
Sao chép
python 05_train_neural_decoder.py
✅ CHECKPOINTS:
Console Output:
Sao chép
================================================================================
NEURAL DECODER TRAINING PIPELINE
================================================================================

[1/6] Loading data...
✓ Train: 46,736 samples
✓ Test:  11,685 samples

[2/6] Encoding symbols...
✓ Encoded 9 unique symbols

[3/6] Building model...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 1, 64)             576       
lstm (LSTM)                 (None, 128)               98816     
dense (Dense)               (None, 64)                8256      
dense_1 (Dense)             (None, 5)                 325       
=================================================================
Total params: 108,973
Trainable params: 108,973

[4/6] Training model...
Epoch 1/50
1460/1460 [==============================] - 12s - loss: 0.0234 - val_loss: 0.0156
Epoch 2/50
1460/1460 [==============================] - 11s - loss: 0.0145 - val_loss: 0.0132
...
Epoch 50/50
1460/1460 [==============================] - 11s - loss: 0.0089 - val_loss: 0.0091

✓ Training complete (best epoch: 47)

[5/6] Evaluating model...
Test Loss: 0.0091

Per-Variable RMSE:
  soil_moisture: 3.45
  pH:            0.23
  nitrogen:      8.67
  temperature:   1.89
  humidity:      4.12

Per-Variable MAE:
  soil_moisture: 2.78
  pH:            0.18
  nitrogen:      6.89
  temperature:   1.45
  humidity:      3.34

[6/6] Saving results...
✓ Model saved to models/neural_decoder.h5
✓ Results saved to results/reports/neural_decoder_results.json

================================================================================
TRAINING COMPLETE
================================================================================
Test RMSE: 4.23
Test MAE:  3.12
Files Created:
Sao chép
models/
└── neural_decoder.h5                    # ✅ Trained model

results/
├── figures/
│   ├── training_history.png            # ✅ Loss curves
│   ├── reconstruction_error.png        # ✅ Error distribution
│   └── prediction_vs_actual.png        # ✅ Scatter plots
└── reports/
    └── neural_decoder_results.json     # ✅ Metrics
Kiểm tra:
Sao chép
# Check model file
ls -lh models/neural_decoder.h5

# View results
cat results/reports/neural_decoder_results.json | python -m json.tool

# Check RMSE
python -c "
import json
with open('results/reports/neural_decoder_results.json') as f:
    data = json.load(f)
    print(f\"Overall RMSE: {data['test_rmse']:.2f}\")
    for var, rmse in data['per_variable_rmse'].items():
        print(f\"  {var}: {rmse:.2f}\")
"
Expected Results (so với paper Section IV.E):
Metric	Expected	Your Result	✅/❌
Overall RMSE	4.2 ± 0.3	?	
Overall MAE	3.1 ± 0.2	?	
Moisture RMSE	3.4 ± 0.2	?	
pH RMSE	0.23 ± 0.05	?	
Nitrogen RMSE	8.6 ± 0.5	?	
Temperature RMSE	1.9 ± 0.2	?	
Humidity RMSE	4.1 ± 0.3	?	
Visual Checks:
Sao chép
open results/figures/training_history.png
Training curves should show:

Loss decreasing smoothly
No overfitting (train/val loss similar)
Convergence around epoch 40-50
🔥 STEP 6: ABLATION STUDY (Bayesian Optimization)
Mục đích:
Optimize membership function parameters
Compare baseline vs optimized
Validate improvement
⚠️ WARNING: This step takes 2-4 hours to run!
Chạy:
Sao chép
python 06_ablation_study.py
✅ CHECKPOINTS:
Console Output:
Sao chép
================================================================================
ABLATION STUDY: BAYESIAN OPTIMIZATION
================================================================================

[1/5] Loading validation data...
✓ Loaded 5,842 validation samples (10% of train)

[2/5] Evaluating baseline...
✓ Baseline accuracy: 88.73%

[3/5] Running Bayesian Optimization...
Iteration 1/50: Current best = -0.8873
Iteration 2/50: Current best = -0.8891
Iteration 3/50: Current best = -0.8912
...
Iteration 50/50: Current best = -0.9045

✓ Optimization complete

[4/5] Evaluating optimized system...
✓ Optimized accuracy: 90.45%

[5/5] Saving results...
✓ Best params saved to results/reports/bo_best_params.json
✓ Optimization history saved to results/reports/bo_history.csv

================================================================================
ABLATION STUDY COMPLETE
================================================================================
Baseline Accuracy:   88.73%
Optimized Accuracy:  90.45%
Improvement:         +1.72%
Files Created:
Sao chép
results/
├── figures/
│   ├── bo_convergence.png              # ✅ Optimization curve
│   └── bo_parameter_importance.png     # ✅ Feature importance
└── reports/
    ├── bo_best_params.json             # ✅ Optimized params
    ├── bo_history.csv                  # ✅ All iterations
    └── ablation_study_report.txt       # ✅ Summary
Kiểm tra:
Sao chép
# View best params
cat results/reports/bo_best_params.json | python -m json.tool

# View improvement
python -c "
import json
with open('results/reports/bo_best_params.json') as f:
    data = json.load(f)
    print(f\"Baseline:  {data['baseline_accuracy']*100:.2f}%\")
    print(f\"Optimized: {data['optimized_accuracy']*100:.2f}%\")
    print(f\"Improvement: {data['improvement']*100:+.2f}%\")
"

# Check convergence
tail -20 results/reports/bo_history.csv
Expected Results (so với paper Section IV.F):
Metric	Expected	Your Result	✅/❌
Baseline Accuracy	88.7%	?	
Optimized Accuracy	90.0-91.0%	?	
Improvement	+1.5-2.5%	?	
Convergence	< 50 iterations	?	
🔥 STEP 7: GENERATE FINAL REPORT
Mục đích:
Tổng hợp tất cả kết quả
So sánh với paper
Generate publication-ready figures
Chạy:
Sao chép
python 07_generate_report.py  # (Nếu có script này)
# HOẶC tự tổng hợp:
Manual Report Generation:
Sao chép
# create_final_report.py
import json
import pandas as pd

print("="*80)
print("FUZSEMCOM FINAL RESULTS SUMMARY")
print("="*80)

# Load all results
with open('results/reports/fse_evaluation_results.json') as f:
    fse_results = json.load(f)

with open('results/reports/neural_decoder_results.json') as f:
    decoder_results = json.load(f)

with open('results/reports/bo_best_params.json') as f:
    bo_results = json.load(f)

# Print summary
print("\n1. FUZZY SEMANTIC ENCODER (FSE)")
print("-"*80)
print(f"Accuracy:        {fse_results['accuracy']*100:.2f}%")
print(f"Expected (paper): 88.70%")
print(f"Difference:      {(fse_results['accuracy']-0.887)*100:+.2f}%")

print("\n2. NEURAL DECODER")
print("-"*80)
print(f"Overall RMSE:    {decoder_results['test_rmse']:.2f}")
print(f"Overall MAE:     {decoder_results['test_mae']:.2f}")
print(f"Expected RMSE:   4.2 ± 0.3")

print("\n3. ABLATION STUDY")
print("-"*80)
print(f"Baseline:        {bo_results['baseline_accuracy']*100:.2f}%")
print(f"Optimized:       {bo_results['optimized_accuracy']*100:.2f}%")
print(f"Improvement:     {bo_results['improvement']*100:+.2f}%")

print("\n" + "="*80)
print("✅ ALL EXPERIMENTS COMPLETE")
print("="*80)
Sao chép
python create_final_report.py
📊 FINAL CHECKPOINT TABLE
So sánh với Paper:
Metric	Paper	Your Result	Status
Section IV.D: FSE Accuracy			
Overall Accuracy	88.7%	?	✅/❌
Optimal Precision	92%	?	✅/❌
Mean Confidence	185/255	?	✅/❌
Section IV.E: Neural Decoder			
Overall RMSE	4.2	?	✅/❌
Overall MAE	3.1	?	✅/❌
Moisture RMSE	3.4	?	✅/❌
pH RMSE	0.23	?	✅/❌
Section IV.F: Ablation Study			
Baseline Accuracy	88.7%	?	✅/❌
Optimized Accuracy	90.0-91.0%	?	✅/❌
Improvement	+1.5-2.5%	?	✅/❌
🎯 QUICK VERIFICATION SCRIPT
Tạo file verify_all_results.py:

Sao chép
"""
verify_all_results.py - Quick verification of all experiments
"""

import json
from pathlib import Path

def check_file(path, description):
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} NOT FOUND")
        return False

def check_metric(value, expected_min, expected_max, name):
    if expected_min <= value <= expected_max:
        print(f"✅ {name}: {value:.2f} (within [{expected_min}, {expected_max}])")
        return True
    else:
        print(f"❌ {name}: {value:.2f} (outside [{expected_min}, {expected_max}])")
        return False

print("="*80)
print("FUZSEMCOM RESULTS VERIFICATION")
print("="*80)

all_pass = True

# Check files
print("\n1. CHECKING FILES...")
all_pass &= check_file('data/processed/semantic_dataset_train.csv', 'Train data')
all_pass &= check_file('data/processed/semantic_dataset_test.csv', 'Test data')
all_pass &= check_file('results/reports/fse_evaluation_results.json', 'FSE results')
all_pass &= check_file('results/reports/neural_decoder_results.json', 'Decoder results')
all_pass &= check_file('models/neural_decoder.h5', 'Trained model')

# Check FSE metrics
print("\n2. CHECKING FSE METRICS...")
try:
    with open('results/reports/fse_evaluation_results.json') as f:
        fse = json.load(f)
    all_pass &= check_metric(fse['accuracy']*100, 87.0, 90.0, 'FSE Accuracy')
    all_pass &= check_metric(fse['confidence_statistics']['mean'], 170, 200, 'Mean Confidence')
except Exception as e:
    print(f"❌ Error loading FSE results: {e}")
    all_pass = False

# Check Decoder metrics
print("\n3. CHECKING DECODER METRICS...")
try:
    with open('results/reports/neural_decoder_results.json') as f:
        decoder = json.load(f)
    all_pass &= check_metric(decoder['test_rmse'], 3.5, 5.0, 'Overall RMSE')
    all_pass &= check_metric(decoder['test_mae'], 2.5, 4.0, 'Overall MAE')
except Exception as e:
    print(f"❌ Error loading decoder results: {e}")
    all_pass = False

# Final verdict
print("\n" + "="*80)
if all_pass:
    print("✅ ALL CHECKS PASSED - RESULTS MATCH PAPER")
else:
    print("❌ SOME CHECKS FAILED - REVIEW ABOVE")
print("="*80)
Sao chép
python verify_all_results.py
🚨 TROUBLESHOOTING
Common Issues:
1. Import Error:
Sao chép
ModuleNotFoundError: No module named 'skfuzzy'
Fix:

Sao chép
pip install scikit-fuzzy
2. File Not Found:
Sao chép
FileNotFoundError: data/raw/Agriculture_dataset_with_metadata.csv
Fix:

Sao chép
# Đảm bảo file CSV ở đúng vị trí
ls data/raw/
3. Low Accuracy (<85%):
Possible causes:

Wrong column mapping
Missing data preprocessing
Incorrect fuzzy rules
Debug:

Sao chép
# Check label distribution
import pandas as pd
df = pd.read_csv('data/processed/semantic_dataset_train.csv')
print(df['semantic_label'].value_counts(normalize=True))
4. High RMSE (>6.0):
Possible causes:

Insufficient training epochs
Wrong normalization
Model architecture issues
Debug:

Sao chép
# Check training history
import json
with open('results/reports/neural_decoder_results.json') as f:
    data = json.load(f)
    print("Training epochs:", data.get('epochs_trained'))
    print("Best epoch:", data.get('best_epoch'))
✅ SUCCESS CRITERIA
Bạn đã hoàn thành thành công khi:

✅ All 7 steps run without errors
✅ FSE accuracy: 87-90%
✅ Decoder RMSE: 3.5-5.0
✅ BO improvement: +1-3%
✅ All figures generated
✅ All reports created
✅ Results match paper (±2%)

