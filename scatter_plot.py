import sys
from load import load
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_numeric_cols, safe_pair


def pearson_corr_algo(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation algorithm for seeing which two features are the most similar"""
    size = x.size
    if size < 2:
        return 0.0

    mx = float(np.sum(x) / size)
    my = float(np.sum(y) / size)
    dx = x - mx
    dy = y - my

    num = float(np.sum(dx * dy))
    den = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))

    if den == 0:
        return 0.0

    return num / den


def draw_scatter(data: pd.DataFrame, subject1: str, subject2: str, houses: List[str]) -> None:
    """function to draw the scatter plot"""
    plt.figure(figsize=(14, 6))
    for h in houses:
        mask = (data["Hogwarts House"] == h)
        x, y = safe_pair(data.loc[mask, subject1], data.loc[mask, subject2])
        plt.scatter(x, y, alpha=0.5, s=60, label=h, edgecolors="black")

    plt.title(f"Scatter plot - {subject1} vs {subject2}\nMost similar feature by Pearson correlation",
              fontsize=14, fontweight="bold")
    plt.xlabel(subject1, fontsize=12)
    plt.ylabel(subject2, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute(data: pd.DataFrame, houses: List):
    """function that splits values by house in order to show
    'What are the two features that are similar?'"""

    for col in data.columns:
        if data[col].dtype not in ['int64', 'float64']:
            continue
        col_data = data[col].dropna()

    subjects = get_numeric_cols(data)
    best_pair = None
    best_corr = 0.0
    best_abs_corr = -1.0

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            x, y = safe_pair(data[s1], data[s2])
            corr = pearson_corr_algo(x, y)
            abs_corr = abs(corr)
            if abs_corr > best_abs_corr:
                best_abs_corr = abs_corr
                best_corr = corr
                best_pair = (s1, s2)

    s1, s2 = best_pair
    print(f"Best correlation between subjects: {s1} and {s2}")
    print(f"Pearson correlation score: {best_corr:.6f}")

    draw_scatter(data, s1, s2, houses)


def main():
    """main func, sorts houses, computes and draws scatter plot"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide exactly one argument: the path to the CSV file.")
        path = sys.argv[1]
        data = load(path)
        houses = sorted([h for h in data["Hogwarts House"].dropna().unique()])
        compute(data, houses)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
