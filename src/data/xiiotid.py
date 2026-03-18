from pathlib import Path
import pandas as pd


def load_xiiotid_dataset(path: str):
    data_path = Path(path)

    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")

    dfs = []
    for file in csv_files:
        print(f"Loading: {file}")
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print("Dataset loaded successfully")
    print("Shape:", combined_df.shape)
    print("Columns:", combined_df.columns.tolist())

    return combined_df