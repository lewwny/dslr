import pandas as pd
import os
from typing import Dict, List, Tuple
import numpy as np
import json

def is_numeric_series(serie) -> bool:
    """checks if a series is numerical"""
    data = str(getattr(serie, "dtype", ""))
    return ("int" in data) or ("float" in data)


def get_numeric_cols(data: pd.DataFrame) -> List[str]:
    """gets numerical columns from data"""
    cols = []
    for col in data.columns:
        if col in ["Hogwarts House", "Index"]:
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
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    return x_arr[mask], y_arr[mask]


def arr_tofloat(arr: np.ndarray) -> np.ndarray:
    """converts an npndarray to a float array"""
    return (np.ndarray(arr, dtype=float))


def sigmoid(arr: np.ndarray) -> np.ndarray:
    """sigmoid function for squashing (takes input and maps to binary)"""
    arr = np.clip(arr, -500, 500)
    result = 1.0 / (1.0 + np.exp(-arr))
    return result


def load_model(path: str) -> dict[str, any]:
    """loads thetas from a json file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        with open(path, 'r') as file:
            model = json.load(file)
            required_keys = ["thetas_dict", "houses", "subjects", "mu", "sigma"]
            for key in required_keys:
                if key not in model:
                    raise ValueError(f"Model missing required key: {key}")

        return model

    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: could not read {path}: {e}")
        return None


def preprocess_vals(data: pd.DataFrame, subjects: List[str], mu: List[float], sigma: List[float]) -> np.ndarray:
    """preprocess vals to remove nan vals that can break the programs"""
    cols = []
    for s in subjects:
        cols.append(arr_tofloat(data[s]))

    raw_vals = np.column_stack(cols)
    vals = raw_vals.copy()

    for i in range (vals.shape[1]):
        mean = float(mu[i])
        std = 1.0
        if float(sigma[i]) != 0.0:
            std = float(sigma[i])
        nan_mask = np.isnan(vals[:, i])
        if np.any(nan_mask):
            vals[nan_mask, i] = mean
        vals[:, i] = (vals[:, i] - mean) / std
    return vals


def add_bias(arr: np.ndarray) -> np.ndarray:
    """adds bias to values of arr"""
    ones = np.ones((arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, arr], axis = 1)
