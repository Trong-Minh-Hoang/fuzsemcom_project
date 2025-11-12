import pandas as pd

from ground_truth import GroundTruthGenerator, save_ground_truth

INPUT_FILE = "processed_data.csv"
GROUND_TRUTH_FILE = "ground_truth_labels.csv"
SEMANTIC_DATASET_FILE = "semantic_dataset.csv"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    generator = GroundTruthGenerator()
    labels = generator.generate(df)
    df_labeled = save_ground_truth(df, labels, GROUND_TRUTH_FILE)
    df_labeled.to_csv(SEMANTIC_DATASET_FILE, index=False)
    print(f"Labeled samples: {len(df_labeled)} saved to {GROUND_TRUTH_FILE}.")
    print(f"Semantic dataset saved to {SEMANTIC_DATASET_FILE}.")
    dropped = len(df) - len(df_labeled)
    if dropped:
        print(f"Dropped {dropped} samples with ground_truth = -1.")


if __name__ == "__main__":
    main()

