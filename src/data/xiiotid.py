import pandas as pd
from pathlib import Path

def load_xiiotid_dataset(path):

    files = list(Path(path).glob("*.csv"))

    if len(files) == 0:
        raise ValueError("No CSV files found")

    dfs = []

    for f in files:
        print("Loading:", f)
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)

    print("Dataset shape:", df.shape)

    return df