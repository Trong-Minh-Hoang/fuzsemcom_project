import pandas as pd

INPUT_FILE = "Agriculture_dataset_with_metadata.csv"
OUTPUT_FILE = "processed_data.csv"
REQUIRED_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df_clean = df.dropna(subset=REQUIRED_COLUMNS)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_clean)} samples to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()

