import json
import os
from datetime import datetime
from typing import Dict, Optional

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


class LDeepSCSimulator:
    """Mô phỏng hiệu năng L-DeepSC dựa trên số liệu trích từ paper."""

    def __init__(self) -> None:
        self.payload_bytes_per_sample = 32  # 32 chiều, 8-bit
        self.base_accuracy = 0.921
        self.inference_time_ms = 15.6
        self.model_size_mb = 2.4
        self.training_time_hours = 4.5
        self.energy_per_bit_uJ = 0.5  # µJ per bit
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

    def simulate_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Sinh dự đoán giả lập theo phân bố nhầm lẫn thường gặp."""
        np.random.seed(42)
        y_true = df["ground_truth"].to_numpy()
        predictions: list[int] = []

        class_accuracy = {
            0: 0.95,
            1: 0.88,
            2: 0.90,
            3: 0.85,
            4: 0.87,
            5: 0.92,
            6: 0.91,
            7: 0.89,
        }
        confusion_map = {
            0: [1, 5, 6],
            1: [0, 2],
            2: [7, 1],
            3: [5, 4],
            4: [6, 3],
            5: [3, 0],
            6: [4, 0],
            7: [2, 0],
        }

        for true_label in y_true:
            acc = class_accuracy.get(int(true_label), self.base_accuracy)
            if np.random.random() < acc:
                predictions.append(int(true_label))
            else:
                errors = confusion_map.get(int(true_label), list(range(len(self.class_names))))
                errors = [e for e in errors if e != int(true_label)]
                if not errors:
                    errors = [e for e in range(len(self.class_names)) if e != int(true_label)]
                predictions.append(int(np.random.choice(errors)))
        return np.asarray(predictions, dtype=int)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            labels=list(range(len(self.class_names))),
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "payload_bytes_per_sample": self.payload_bytes_per_sample,
            "inference_time_ms": self.inference_time_ms,
            "model_size_mb": self.model_size_mb,
            "training_time_hours": self.training_time_hours,
            "class_report": report,
        }

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "deepsc_confusion_matrix.png",
    ) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("L-DeepSC - Confusion Matrix (Simulated)")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


class ComparisonAnalyzer:
    def __init__(self) -> None:
        self.deepsc_sim = LDeepSCSimulator()
        self.comparison_results: Dict[str, object] = {}

    def run_comprehensive_comparison(self, df: pd.DataFrame) -> Optional[Dict[str, object]]:
        fse_results = self._load_fse_results()
        if fse_results is None:
            return None

        print("\nFUZSEMCOM vs L-DeepSC COMPREHENSIVE COMPARISON")
        print("=" * 70)
        print("Simulating L-DeepSC performance ...")
        predictions = self.deepsc_sim.simulate_predictions(df)
        deepsc_results = self.deepsc_sim.calculate_metrics(df["ground_truth"].to_numpy(), predictions)
        print("L-DeepSC simulation completed.")

        self.deepsc_sim.plot_confusion_matrix(
            df["ground_truth"].to_numpy(),
            predictions,
            save_path="deepsc_confusion_matrix.png",
        )
        self._save_deepsc_classification_report(deepsc_results["class_report"])  # type: ignore[arg-type]

        self.comparison_results = self._calculate_comparison_metrics(fse_results, deepsc_results, df)
        self._print_detailed_comparison()
        self._create_comprehensive_plots(fse_results, deepsc_results)
        self._generate_comparison_report(fse_results, deepsc_results)

        with open("deepsc_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(self.comparison_results, f, indent=2)

        print("\nComparison completed successfully!")
        print("Results saved to 'deepsc_comparison_results.json'")
        return self.comparison_results

    def _load_fse_results(self) -> Optional[Dict[str, object]]:
        try:
            with open("fse_evaluation_results.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            print("Loaded FuzSemCom evaluation results.")
            return data
        except FileNotFoundError:
            print("Error: fse_evaluation_results.json not found. Hãy chạy 05_evaluate_fse.py trước.")
            return None

    def _save_deepsc_classification_report(self, report: Dict[str, Dict[str, float]]) -> None:
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("deepsc_classification_report.csv")

    def _calculate_comparison_metrics(
        self,
        fse_results: Dict[str, float],
        deepsc_results: Dict[str, float],
        df: pd.DataFrame,
    ) -> Dict[str, object]:
        n_samples = len(df)
        fse_payload = n_samples * 1  # 1 byte biểu tượng
        deepsc_payload = n_samples * deepsc_results["payload_bytes_per_sample"]
        bandwidth_saving = (deepsc_payload - fse_payload) / deepsc_payload * 100

        energy_per_bit_j = 0.5e-6  # 0.5 µJ
        fse_energy_total = fse_payload * 8 * energy_per_bit_j
        deepsc_energy_total = deepsc_payload * 8 * energy_per_bit_j
        energy_saving = (deepsc_energy_total - fse_energy_total) / deepsc_energy_total * 100

        accuracy_diff = (deepsc_results["accuracy"] - fse_results["accuracy"]) * 100
        f1_diff = (deepsc_results["f1_macro"] - fse_results["f1_macro"]) * 100

        payload_reduction_ratio = deepsc_payload / max(fse_payload, 1)
        fse_inference_ms = fse_results.get("avg_prediction_time", 0.0) * 1000
        speed_advantage = (
            deepsc_results["inference_time_ms"] / max(fse_inference_ms, 1e-6)
            if fse_inference_ms
            else float("inf")
        )

        devices = 1000
        messages_per_day = 24
        days_per_year = 365
        annual_fse_payload = devices * messages_per_day * days_per_year * 1
        annual_deepsc_payload = devices * messages_per_day * days_per_year * deepsc_results["payload_bytes_per_sample"]
        annual_bandwidth_saved = annual_deepsc_payload - annual_fse_payload

        return {
            "performance_comparison": {
                "fuzsemcom": {
                    "accuracy": fse_results["accuracy"],
                    "f1_macro": fse_results["f1_macro"],
                    "f1_weighted": fse_results["f1_weighted"],
                    "precision_macro": fse_results["precision_macro"],
                    "recall_macro": fse_results["recall_macro"],
                },
                "l_deepsc": {
                    "accuracy": deepsc_results["accuracy"],
                    "f1_macro": deepsc_results["f1_macro"],
                    "f1_weighted": deepsc_results["f1_weighted"],
                    "precision_macro": deepsc_results["precision_macro"],
                    "recall_macro": deepsc_results["recall_macro"],
                },
                "differences": {
                    "accuracy_diff_percent": accuracy_diff,
                    "f1_diff_percent": f1_diff,
                },
            },
            "efficiency_comparison": {
                "communication": {
                    "fse_payload_per_sample": 1,
                    "deepsc_payload_per_sample": deepsc_results["payload_bytes_per_sample"],
                    "bandwidth_saving_percent": bandwidth_saving,
                    "payload_reduction_ratio": payload_reduction_ratio,
                },
                "computational": {
                    "fse_inference_time_ms": fse_inference_ms,
                    "deepsc_inference_time_ms": deepsc_results["inference_time_ms"],
                    "speed_advantage_ratio": speed_advantage,
                },
                "energy": {
                    "fse_energy_per_message_uJ": fse_energy_total * 1e6 / n_samples,
                    "deepsc_energy_per_message_uJ": deepsc_energy_total * 1e6 / n_samples,
                    "energy_saving_percent": energy_saving,
                },
            },
            "deployment_comparison": {
                "fuzsemcom": {
                    "hardware_requirement": "ESP32, Arduino (≈$2-5)",
                    "model_size": "Rule-based (vài KB)",
                    "training_requirement": "Không cần",
                    "interpretability": "Cao (fuzzy rules)",
                    "deployment_complexity": "Thấp",
                    "maintenance": "Cập nhật rule thủ công",
                },
                "l_deepsc": {
                    "hardware_requirement": "Edge compute (≈$50-200)",
                    "model_size": f"{deepsc_results['model_size_mb']} MB",
                    "training_requirement": f"{deepsc_results['training_time_hours']} giờ fine-tune",
                    "interpretability": "Thấp (black-box)",
                    "deployment_complexity": "Cao",
                    "maintenance": "Cần retraining định kỳ",
                },
            },
            "scalability_analysis": {
                "annual_bandwidth_saved_gb": annual_bandwidth_saved / (1024**3),
                "devices_supported": devices,
                "messages_per_year": devices * messages_per_day * days_per_year,
                "cost_savings_estimate": {
                    "bandwidth_cost_saved_usd": annual_bandwidth_saved * 0.10 / 1024,  # $0.10/MB
                    "hardware_cost_difference_usd": devices * 45,
                },
            },
            "use_case_recommendations": {
                "fuzsemcom_preferred": [
                    "Cảm biến chạy pin, yêu cầu tuổi thọ dài",
                    "Trang trại vùng xa, băng thông hạn chế",
                    "Ưu tiên ra quyết định dễ giải thích",
                    "Chi phí phần cứng phải thấp",
                ],
                "l_deepsc_preferred": [
                    "Ứng dụng đòi hỏi độ chính xác tối đa",
                    "Có sẵn edge server cấu hình mạnh",
                    "Nhận dạng mẫu phức tạp, dữ liệu phong phú",
                    "Chấp nhận mô hình black-box",
                ],
            },
        }

    def _print_detailed_comparison(self) -> None:
        results = self.comparison_results
        perf = results["performance_comparison"]
        eff = results["efficiency_comparison"]
        deploy = results["deployment_comparison"]
        scale = results["scalability_analysis"]
        use_cases = results["use_case_recommendations"]

        print("\nPERFORMANCE METRICS COMPARISON")
        print("=" * 60)
        print(f"{'Metric':20}{'FuzSemCom':>15}{'L-DeepSC':>15}{'Diff (%)':>12}")
        print("-" * 62)
        print(
            f"{'Accuracy':20}"
            f"{perf['fuzsemcom']['accuracy']:>15.3f}"
            f"{perf['l_deepsc']['accuracy']:>15.3f}"
            f"{perf['differences']['accuracy_diff_percent']:>12.1f}"
        )
        print(
            f"{'F1 (Macro)':20}"
            f"{perf['fuzsemcom']['f1_macro']:>15.3f}"
            f"{perf['l_deepsc']['f1_macro']:>15.3f}"
            f"{perf['differences']['f1_diff_percent']:>12.1f}"
        )
        print(
            f"{'Precision':20}"
            f"{perf['fuzsemcom']['precision_macro']:>15.3f}"
            f"{perf['l_deepsc']['precision_macro']:>15.3f}"
            f"{'':>12}"
        )
        print(
            f"{'Recall':20}"
            f"{perf['fuzsemcom']['recall_macro']:>15.3f}"
            f"{perf['l_deepsc']['recall_macro']:>15.3f}"
            f"{'':>12}"
        )

        print("\nEFFICIENCY METRICS COMPARISON")
        print("=" * 60)
        comm = eff["communication"]
        comp = eff["computational"]
        energy = eff["energy"]
        print(
            f"Payload Size: FuzSemCom {comm['fse_payload_per_sample']} byte | "
            f"L-DeepSC {comm['deepsc_payload_per_sample']} byte"
        )
        print(f"Bandwidth Saving: {comm['bandwidth_saving_percent']:.1f}%")
        print(f"Payload Reduction Ratio: {comm['payload_reduction_ratio']:.1f}:1")
        print(
            f"Inference Time: FuzSemCom {comp['fse_inference_time_ms']:.3f} ms | "
            f"L-DeepSC {comp['deepsc_inference_time_ms']:.1f} ms"
        )
        print(f"Speed Advantage (L-DeepSC / FSE): {comp['speed_advantage_ratio']:.1f}x")
        print(
            "Energy per Message: "
            f"FuzSemCom {energy['fse_energy_per_message_uJ']:.2f} µJ | "
            f"L-DeepSC {energy['deepsc_energy_per_message_uJ']:.2f} µJ"
        )
        print(f"Energy Saving: {energy['energy_saving_percent']:.1f}%")

        print("\nDEPLOYMENT COMPARISON")
        print("=" * 60)
        print("FuzSemCom:")
        for k, v in deploy["fuzsemcom"].items():
            print(f"  - {k.replace('_', ' ').title()}: {v}")
        print("\nL-DeepSC:")
        for k, v in deploy["l_deepsc"].items():
            print(f"  - {k.replace('_', ' ').title()}: {v}")

        print("\nSCALABILITY & COST ANALYSIS")
        print("=" * 60)
        print(f"Annual Bandwidth Saved: {scale['annual_bandwidth_saved_gb']:.2f} GB")
        print(
            "Bandwidth Cost Savings (est.): "
            f"${scale['cost_savings_estimate']['bandwidth_cost_saved_usd']:.2f}/năm"
        )
        print(
            "Hardware Cost Difference (est.): "
            f"${scale['cost_savings_estimate']['hardware_cost_difference_usd']:,}"
        )

        print("\nUSE CASE RECOMMENDATIONS")
        print("=" * 60)
        print("\nFuzSemCom phù hợp khi:")
        for case in use_cases["fuzsemcom_preferred"]:
            print(f"  • {case}")
        print("\nL-DeepSC phù hợp khi:")
        for case in use_cases["l_deepsc_preferred"]:
            print(f"  • {case}")

    def _create_comprehensive_plots(
        self,
        fse_results: Dict[str, float],
        deepsc_results: Dict[str, float],
    ) -> None:
        plt.style.use("default")
        sns.set_palette("Set2")

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("FuzSemCom vs L-DeepSC: Comparison Overview", fontsize=18, fontweight="bold")

        metrics = ["Accuracy", "F1 (Macro)", "Precision", "Recall"]
        fse_perf = [
            fse_results["accuracy"],
            fse_results["f1_macro"],
            fse_results["precision_macro"],
            fse_results["recall_macro"],
        ]
        deepsc_perf = [
            deepsc_results["accuracy"],
            deepsc_results["f1_macro"],
            deepsc_results["precision_macro"],
            deepsc_results["recall_macro"],
        ]

        ax1 = fig.add_subplot(2, 2, 1)
        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width / 2, fse_perf, width, label="FuzSemCom", color="#2E8B57")
        ax1.bar(x + width / 2, deepsc_perf, width, label="L-DeepSC", color="#4169E1")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=30)
        ax1.set_ylabel("Score")
        ax1.set_title("Performance Metrics")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        comm = self.comparison_results["efficiency_comparison"]["communication"]
        ax2.bar(["FuzSemCom", "L-DeepSC"], [comm["fse_payload_per_sample"], comm["deepsc_payload_per_sample"]])
        ax2.set_title("Payload per Sample (bytes)")
        ax2.grid(axis="y", alpha=0.3)

        ax3 = fig.add_subplot(2, 2, 3)
        comp = self.comparison_results["efficiency_comparison"]["computational"]
        ax3.bar(
            ["FuzSemCom", "L-DeepSC"],
            [comp["fse_inference_time_ms"], comp["deepsc_inference_time_ms"]],
            color=["#2E8B57", "#4169E1"],
        )
        ax3.set_ylabel("Time (ms)")
        ax3.set_title("Inference Time")
        ax3.grid(axis="y", alpha=0.3)

        ax4 = fig.add_subplot(2, 2, 4)
        energy = self.comparison_results["efficiency_comparison"]["energy"]
        ax4.bar(
            ["FuzSemCom", "L-DeepSC"],
            [energy["fse_energy_per_message_uJ"], energy["deepsc_energy_per_message_uJ"]],
            color=["#2E8B57", "#4169E1"],
        )
        ax4.set_ylabel("µJ per message")
        ax4.set_title("Transmission Energy")
        ax4.grid(axis="y", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig("deepsc_comparison_overview.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _generate_comparison_report(
        self,
        fse_results: Dict[str, float],
        deepsc_results: Dict[str, float],
    ) -> None:
        rows = [
            {
                "system": "FuzSemCom",
                "accuracy": fse_results["accuracy"],
                "f1_macro": fse_results["f1_macro"],
                "precision_macro": fse_results["precision_macro"],
                "recall_macro": fse_results["recall_macro"],
                "payload_bytes": 1,
                "inference_time_ms": fse_results.get("avg_prediction_time", 0.0) * 1000,
                "energy_per_message_uJ": self.comparison_results["efficiency_comparison"]["energy"][
                    "fse_energy_per_message_uJ"
                ],
            },
            {
                "system": "L-DeepSC (simulated)",
                "accuracy": deepsc_results["accuracy"],
                "f1_macro": deepsc_results["f1_macro"],
                "precision_macro": deepsc_results["precision_macro"],
                "recall_macro": deepsc_results["recall_macro"],
                "payload_bytes": deepsc_results["payload_bytes_per_sample"],
                "inference_time_ms": deepsc_results["inference_time_ms"],
                "energy_per_message_uJ": self.comparison_results["efficiency_comparison"]["energy"][
                    "deepsc_energy_per_message_uJ"
                ],
            },
        ]
        pd.DataFrame(rows).to_csv("deepsc_comparison_summary.csv", index=False)


def load_semantic_dataset(project_root: str) -> pd.DataFrame:
    semantic_path = os.path.join(project_root, "semantic_dataset.csv")
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(
            "Không tìm thấy semantic_dataset.csv. Hãy chạy 04_ground_truth.py để tạo dữ liệu."
        )
    df = pd.read_csv(semantic_path)
    if "ground_truth" not in df.columns:
        raise ValueError("semantic_dataset.csv thiếu cột 'ground_truth'.")
    return df


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    df = load_semantic_dataset(project_root)
    analyzer = ComparisonAnalyzer()
    results = analyzer.run_comprehensive_comparison(df)
    if results is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nHoàn tất so sánh lúc {timestamp}.")


if __name__ == "__main__":
    main()

