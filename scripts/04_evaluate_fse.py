import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fuzzy_system import FSEPrediction, TomatoFuzzySystem  # noqa: E402

SEMANTIC_DATASET = PROJECT_ROOT / "data" / "processed" / "semantic_dataset.csv"


@dataclass
class FSEEvaluator:
    fuzzy_system: TomatoFuzzySystem
    enable_thresholds: bool = True
    enable_fallback: bool = True

    def __post_init__(self) -> None:
        self.class_names = [
            "optimal",
            "nutrient_deficiency",
            "fungal_risk",
            "water_deficit_acidic",
            "water_deficit_alkaline",
            "acidic_soil",
            "alkaline_soil",
            "heat_stress",
        ]

    def evaluate_performance(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        print("Evaluating Fuzzy Semantic Encoder Performance...")
        print("=" * 60)

        predictions: list[int] = []
        confidences: list[float] = []
        rule_strengths: list[Dict[str, float]] = []
        prediction_times: list[float] = []

        for idx, row in df.iterrows():
            start = time.time()
            pred: FSEPrediction = self.fuzzy_system.predict(
                moisture=row["Moisture"],
                ph=row["pH"],
                nitrogen=row["N"],
                temperature=row["Temperature"],
                humidity=row["Humidity"],
                ndi_label=row.get("NDI_Label"),
                pdi_label=row.get("PDI_Label"),
                enable_thresholds=self.enable_thresholds,
                enable_fallback=self.enable_fallback,
            )
            predictions.append(pred.class_id)
            confidences.append(pred.confidence)
            rule_strengths.append(pred.rule_strengths)
            prediction_times.append(time.time() - start)

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")

        df = df.copy()
        df["fse_prediction"] = predictions
        df["fse_confidence"] = confidences
        df["fse_rule_strengths"] = rule_strengths

        y_true = df["ground_truth"].to_numpy()
        y_pred = df["fse_prediction"].to_numpy()
        valid_mask = (y_true >= 0) & (y_pred >= 0) & (y_pred <= 7)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            raise ValueError("Không có mẫu hợp lệ nào sau khi lọc ground_truth/prediction.")

        results = {
            "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
            "f1_macro": float(f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
            ),
            "valid_predictions": int(len(y_true_valid)),
            "total_samples": int(len(df)),
            "prediction_success_rate": float(len(y_true_valid) / len(df)),
            "avg_prediction_time": float(np.mean(prediction_times)),
            "total_prediction_time": float(np.sum(prediction_times)),
            "avg_confidence": float(np.mean(confidences)),
        }

        self._print_results(results)
        self._plot_confusion_matrix(y_true_valid, y_pred_valid)
        self._generate_classification_report(y_true_valid, y_pred_valid)

        return results, df

    def _print_results(self, results: Dict[str, float]) -> None:
        print("\nFUZZY SEMANTIC ENCODER PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"Semantic Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"F1-Score (Macro): {results['f1_macro']:.3f}")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.3f}")
        print(f"Precision (Macro): {results['precision_macro']:.3f}")
        print(f"Recall (Macro): {results['recall_macro']:.3f}")
        print(f"Prediction Success Rate: {results['prediction_success_rate']:.3f}")
        print(f"Average Prediction Time: {results['avg_prediction_time']*1000:.2f} ms")
        print(f"Valid Predictions: {results['valid_predictions']}/{results['total_samples']}")
        if "avg_confidence" in results:
            print(f"Average Confidence: {results['avg_confidence']:.3f}")

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Fuzzy Semantic Encoder - Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("fse_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            labels=range(len(self.class_names)),
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        print("\nDETAILED CLASSIFICATION REPORT")
        print("=" * 80)
        print(report_df)
        report_df.to_csv("fse_classification_report.csv")
        print("\nClassification report saved to 'fse_classification_report.csv'")


def load_semantic_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: {path} not found!")
        print("Please run 04_ground_truth.py first to generate the semantic dataset.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")
    if "ground_truth" not in df.columns:
        raise ValueError("semantic_dataset.csv thiếu cột 'ground_truth'.")
    return df


def main() -> None:
    df = load_semantic_dataset(SEMANTIC_DATASET)
    enable_thresholds = os.getenv("FSE_ENABLE_THRESHOLDS", "1") != "0"
    enable_fallback = os.getenv("FSE_ENABLE_FALLBACK", "1") != "0"

    evaluator = FSEEvaluator(
        TomatoFuzzySystem(),
        enable_thresholds=enable_thresholds,
        enable_fallback=enable_fallback,
    )
    start = datetime.now()
    results, df_with_predictions = evaluator.evaluate_performance(df)

    with open("fse_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df_with_predictions.to_csv("fse_predictions.csv", index=False)
    duration = datetime.now() - start
    print("\nEvaluation complete!")
    print("- fse_evaluation_results.json")
    print("- fse_predictions.csv")
    print("- fse_confusion_matrix.png")
    print("- fse_classification_report.csv")
    print(f"Total runtime: {duration.total_seconds():.2f} seconds")


if __name__ == "__main__":
    main()
