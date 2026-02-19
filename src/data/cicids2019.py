"""Loader for the CICIDS-2019 dataset."""
from pathlib import Path

import pandas as pd


def load_cicids2019(raw_dir: str | Path) -> pd.DataFrame:
    """Load and return raw CICIDS-2019 data as a DataFrame.

    Args:
        raw_dir: Path to data/raw/cicids2019/
    """
    raise NotImplementedError
