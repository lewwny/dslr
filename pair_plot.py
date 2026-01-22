import pandas as pd
from typing import List, Tuple
from utils import get_numeric_cols, safe_pair
import numpy as np
from load import load
import matplotlib.pyplot as plt

def plot_histogram(ax, data: pd.DataFrame, subject: str, houses: List[str]) -> None:
    """draw histogram on the given axis"""
    for h in houses:
        mask = (data["Hogwarts House"] == h)
        vals = data.loc[mask, subject].dropna()
        ax.hist(vals, bins=15, alpha=0.5, label=h)
    ax.grid(True, alpha=0.3)


def plot_scatter(ax, data: pd.DataFrame, subject1: str, subject2: str, houses: List[str]) -> None:
    """draw scatter plot on the given axis"""
    for h in houses:
        mask = (data["Hogwarts House"] == h)
        x, y = safe_pair(data.loc[mask, subject1], data.loc[mask, subject2])
        ax.scatter(x, y, alpha=0.5, s=20, label=h)
    ax.grid(True, alpha=0.3)

def draw_pair_plot(data: pd.DataFrame, houses: List[str]) -> None:
    """function to draw pair plot for the most similar features"""
    subjects = get_numeric_cols(data)
    n = len(subjects)
    if n < 2:
        print("Not enough numeric columns to create pair plot.")
        return
    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(25, 25))
    print(f"Creating pair plot for {n} subjects.")
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            subject_x = subjects[j]
            subject_y = subjects[i]
            if i == j:
                plot_histogram(ax, data, subject_x, houses)
            else:
                plot_scatter(ax, data, subject_x, subject_y, houses)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                label = subject_x.replace(" ", "\n")
                ax.set_xlabel(label, fontsize=9)
            if j == 0:
                label = subject_y.replace(" ", "\n")
                ax.set_ylabel(label, fontsize=9)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to load data and generate pair plots for the most similar features."""
    try:
        data = load("./datasets/dataset_train.csv")
        houses = sorted(data["Hogwarts House"].dropna().unique())
        draw_pair_plot(data, houses)

    except Exception as e:
        print(f"Error loading data: {e}")
        return


if __name__ == "__main__":
    main()