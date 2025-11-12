import pandas as pd


def main() -> None:
    df = pd.read_csv("Agriculture_dataset_with_metadata.csv")
    print("Columns:", df.columns.tolist())
    preview_columns = [
        "Moisture",
        "pH",
        "N",
        "Temperature",
        "Humidity",
        "NDI_Label",
        "PDI_Label",
    ]
    existing = [col for col in preview_columns if col in df.columns]
    print(df[existing].head())


if __name__ == "__main__":
    main()

