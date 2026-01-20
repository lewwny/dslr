import sys
from load import load_csv
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def draw_histogram(data: pd.DataFrame, feature: str, houses: List[str], bins: int = 20) -> None:
    """function to draw the histogram for the courses"""
    plt.figure(figsize=(14, 6))
    for h in houses:
        mask = (data["Hogwarts House"] == h)
        vals = safe_val(data.loc[mask, feature])
        plt.hist(vals, bins=bins, alpha=0.5, label=h, edgecolor="black")
    
    plt.title(f"Histogram - {feature}\nHomogeneous distribution across houses", fontsize=14, fontweight="bold")
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def homogeneous_score(data: pd.DataFrame, feature: str, houses: List[str], bins: int = 20) -> float:
    """calculates homogenity score for houses"""
    values = safe_val(data[feature])
    if values.size < 2:
        return float("inf")
    minval = float(np.min(values))
    maxval = float(np.max(values))
    if minval == maxval:
        return 0.0
    

def compute(data: pd.DataFame, houses: List):
    """function that splits values by house in order to show
    'Which Hogwarts course has a homogeneous score distribution
    between all four houses?'"""

    for col in data.columns:
        if data[col].dtype not in ['int64', 'float64']:
            continue
        col_data = data[col].dropna()

    best_feature = None
    best_score = float("inf")

    for f in features:
        score = homogeneous_score(data, f, houses, bins=20)
        if score < best_score:
            best_score = score

    print(f"Most homogeneous course: {best_feature}\nScore: {best_score:.6f}")




def main():
    """main func"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide exactly one argument: the path to the CSV file.")
        path = sys.argv[1]
        data = load_csv(path)
        houses = sorted([h for h in data.dropna().unique])
        compute(data, houses)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()