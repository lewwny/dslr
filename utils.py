import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

def is_numeric_series(serie) -> bool:
    """checks if a series is numerical"""
    data = str(getattr(serie, "dtype", ""))
    return ("int" in data) or ("float" in data)


def get_numeric_cols(data: pd.DataFrame) -> List[str]:
    """gets numerical columns from data"""
    cols = []
    for col in data.columns:
        if col == "Hogwarts House":
            continue
        if is_numeric_series(data[col]):
            cols.append(col)
    return cols


def safe_val(series) -> np.ndarray:
    """safe val for nans in a series"""
    arr = np.array(series, dtype=float)
    return arr[~np.isnan(arr)]


def safe_pair(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """safe pairs for nans in pairs"""
    return safe_val(x), safe_val(y)
