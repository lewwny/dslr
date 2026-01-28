from load import load
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_numeric_cols, safe_val

def js_algo(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """Jensen Shannon algorithm for calculating probability of distribution score"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl1 = np.sum(p * np.log(p / m))
    kl2 = np.sum(q * np.log(q / m))

    return 0.5 * kl1 + 0.5 * kl2


def draw_histogram(data: pd.DataFrame, subject: str, houses: List[str], bins: int = 20) -> None:
    """function to draw the histogram for the courses"""
    plt.figure(figsize=(14, 6))
    for h in houses:
        mask = (data["Hogwarts House"] == h)
        vals = safe_val(data.loc[mask, subject])
        plt.hist(vals, bins=bins, alpha=0.5, label=h, edgecolor="black")

    plt.title(f"Histogram - {subject}\nHomogeneous distribution across houses", fontsize=14, fontweight="bold")
    plt.xlabel(subject, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def homogeneous_score(data: pd.DataFrame, subject: str, houses: List[str], bins: int = 20) -> float:
    """calculates homogenity score for houses"""
    values = safe_val(data[subject])
    if values.size < 2:
        return float("inf")
    minval = float(np.min(values))
    maxval = float(np.max(values))
    if minval == maxval:
        return 0.0

    edge = np.linspace(minval, maxval, bins + 1)
    hists: Dict[str, np.ndarray] = {}
    for h in houses:
        mask = (data["Hogwarts House"] == h)
        vals = safe_val(data.loc[mask, subject])
        if vals.size == 0:
            return float("inf")
        hist, _ = np.histogram(vals, bins=edge)
        hist = hist.astype(float)
        if hist.sum() == 0:
            return float("inf")
        hists[h] = hist / hist.sum()

    scores = []
    eps = 1e-10
    for i in range(len(houses)):
        for j in range(i + 1, len(houses)):
            scores.append(js_algo(hists[houses[i]], hists[houses[j]], eps))
    return float(np.mean(scores)) if scores else float("inf")


def compute(data: pd.DataFrame, houses: List):
    """function that splits values by house in order to show
    'Which Hogwarts course has a homogeneous score distribution
    between all four houses?'"""

    for col in data.columns:
        if data[col].dtype not in ['int64', 'float64']:
            continue
        col_data = data[col].dropna()

    subjects = get_numeric_cols(data)
    best_subject = None
    best_score = float("inf")

    for s in subjects:
        score = homogeneous_score(data, s, houses, bins=20)
        if score < best_score:
            best_score = score
            best_subject = s

    print(f"Most homogeneous course: {best_subject}\nScore: {best_score:.6f}")

    draw_histogram(data, best_subject, houses, bins=20)


def main():
    """main func, sorts data and calls compute"""
    try:
        path = "./datasets/dataset_train.csv"
        data = load(path)
        houses = sorted([h for h in data["Hogwarts House"].dropna().unique()])
        compute(data, houses)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
