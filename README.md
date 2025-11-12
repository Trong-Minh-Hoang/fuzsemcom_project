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
