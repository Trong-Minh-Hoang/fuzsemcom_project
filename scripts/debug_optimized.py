"""
Debug script để phân tích tại sao accuracy thấp
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from fuzzy_system_optimized import FuzzySemanticEncoderOptimized

DATA_PATH = Path(__file__).resolve().parents[1] / "fuzsemcom_project" / "semantic_dataset.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    encoder = FuzzySemanticEncoderOptimized(use_scikit_fuzzy=False)
    
    # Lấy 100 mẫu đầu để debug
    sample_df = df.head(100)
    
    print("=" * 70)
    print("Debug: Phân tích 100 mẫu đầu tiên")
    print("=" * 70)
    
    predictions = []
    confidences = []
    details = []
    
    for idx, row in sample_df.iterrows():
        semantic_class, confidence, memberships = encoder.encode(
            moisture=float(row["Moisture"]),
            ph=float(row["pH"]),
            nitrogen=float(row["N"]),
            temperature=float(row["Temperature"]),
            humidity=float(row["Humidity"]),
        )
        predictions.append(semantic_class)
        confidences.append(confidence)
        
        details.append({
            'idx': idx,
            'true': int(row['ground_truth']),
            'pred': semantic_class,
            'match': int(row['ground_truth']) == semantic_class,
            'confidence': confidence,
            'moisture': row['Moisture'],
            'ph': row['pH'],
            'nitrogen': row['N'],
            'temp': row['Temperature'],
            'humidity': row['Humidity'],
        })
    
    details_df = pd.DataFrame(details)
    
    # Phân tích
    print(f"\nAccuracy trên 100 mẫu: {(details_df['match'].sum() / len(details_df)):.4f}")
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(details_df['true'], details_df['pred'], labels=range(8))
    print(cm)
    
    print(f"\nPhân bố ground_truth:")
    print(details_df['true'].value_counts().sort_index())
    
    print(f"\nPhân bố predictions:")
    print(details_df['pred'].value_counts().sort_index())
    
    print(f"\nCác mẫu bị sai (top 10):")
    wrong = details_df[~details_df['match']].head(10)
    for _, row in wrong.iterrows():
        print(f"  Sample {row['idx']}: True={row['true']}, Pred={row['pred']}, Conf={row['confidence']:.3f}")
        print(f"    Input: M={row['moisture']:.1f}, pH={row['ph']:.2f}, N={row['nitrogen']:.1f}, T={row['temp']:.1f}, H={row['humidity']:.1f}")
    
    print(f"\nConfidence statistics:")
    print(f"  Mean: {details_df['confidence'].mean():.4f}")
    print(f"  Min: {details_df['confidence'].min():.4f}")
    print(f"  Max: {details_df['confidence'].max():.4f}")
    print(f"  Std: {details_df['confidence'].std():.4f}")
    
    # So sánh với pipeline chính
    print(f"\n" + "=" * 70)
    print("So sánh với kết quả pipeline chính (nếu có):")
    print("=" * 70)
    if 'fse_prediction' in df.columns:
        pipeline_preds = df['fse_prediction'].head(100).values
        pipeline_acc = (pipeline_preds == df['ground_truth'].head(100).values).mean()
        print(f"Pipeline chính accuracy (100 mẫu): {pipeline_acc:.4f}")
        print(f"Optimized accuracy (100 mẫu): {(details_df['match'].sum() / len(details_df)):.4f}")
        
        # So sánh predictions
        matches = (pipeline_preds == predictions).sum()
        print(f"Số predictions giống nhau: {matches}/100")

if __name__ == "__main__":
    main()

