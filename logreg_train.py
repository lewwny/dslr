import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Dict
from load import load
from utils import get_numeric_cols, sigmoid, add_bias

def replace_nan(arr: np.ndarray) -> np.ndarray:
    """replaces nan values with the mean for ML"""
    copy = arr.copy()
    for col_idx in range(copy.shape[1]):
        col = copy[:, col_idx]
        valid_mask = ~np.isnan(col)
        if np.any(valid_mask):
            col_mean = np.mean(col[valid_mask])
        else:
            col_mean = 0.0
        nan_mask = np.isnan(col)
        copy[nan_mask, col_idx] = col_mean

    return copy


def compute_mu_sig(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """computes the mean and std for an array"""
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)
    sigma[sigma == 0] = 1.0

    return mu, sigma


def compute_cost(h: np.ndarray, y: np.ndarray) -> float:
    """computes binary cross entropy (log loss)
    formula: J(θ) = -1/m * sigma[y*log(h) + (1-y)*log(1-h)]
    - When y=1: we want h close to 1, so -log(h) penalizes low h
    - When y=0: we want h close to 0, so -log(1-h) penalizes high h
    - The penalty grows exponentially for confident wrong predictions
    - We clip h to avoid log(0) = -inf for numerical stability"""
    m = len(y)
    eps = 1e-15
    h_clip = np.clip(h, eps, 1 - eps)

    cost = -1/m * np.sum(y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip))
    return cost


def normalize(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z score normalization of arr vals using the mean and std"""
    return (arr - mu) / sigma


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: float,
                     learning_rate: float, iterations: int, verb: bool = True):
    """Batch Gradient Descent for logistic regression.
    1. Compute predictions: h = sigmoid(X @ θ)
    2. Compute gradient: deltaJ = (1/m) * X.T(h - y)
    3. Update weights: θ = θ - alpha * deltaJ
    4. Repeat for n iterations

    - X: Feature matrix with bias column (m samples * n+1 features)
    - y: Binary labels (m * 1), where 1 = this house, 0 = other houses
    - theta: Initial weights (n+1 * 1), usually zeros
    - learning_rate (alpha): Step size: too high -> diverge, too low -> slow
    - iterations: Number of gradient descent steps"""
    m = len(y)
    cost_hist = []

    for i in range(iterations):
        # computes prediction with linear combination + squashing to probas
        z = X @ theta
        h = sigmoid(z)

        # compute gradient from the derivative of cost function
        gradient = (1/m) * (X.T @ (h - y))

        # update weights
        theta - theta - learning_rate * gradient

        # compute cost and add to history
        cost = compute_cost(h, y)
        cost_hist.append(cost)

        # print verbose
        if verb and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{iterations}, Cost: {cost:.6f}")

    return theta, cost_hist


def train_model(X: np.ndarray, y_labels: np.ndarray, houses: list, lr: float, iters: int):
    """train model using logistic regression binary classifiers
    for each house we use:
    - positive samples (y=1) -> students in this house
    - negative samples (y=0) -> students in all other houses
    results in 4 theta vectors -> 1 theta per house
    in prediction, we run all 4 thetas and keep one with highest probability"""
    n_features = X.shape[1]
    thetas_dict = {}
    costs = {}

    for house in houses:
        print(f"Training classifier for {house}")
        y_bin = (y_labels == house).astype(float).reshape(-1, 1)

        n_pos = np.sum(y_bin)
        n_neg = len(y_bin) - n_pos
        print(f"Samples: {int(n_pos)} positive, {int(n_neg)} negative")

        theta = np.zeros((n_features, 1))
        theta, cost_hist = gradient_descent(X, y_bin, theta, lr, iters)

        thetas_dict[house] = theta.flatten().tolist()
        costs[house] = cost_hist
        print(f"Final cost for {house}: {cost_hist[-1]:.6f}")

    return thetas_dict, costs


def plot_training_metrics(cost_history: dict, iterations: int) -> None:
    """visualize training progress: cost over iterations"""
    plt.figure(figsize=(12, 5))

    # linear scale
    plt.subplot(1, 2, 1)
    for house, costs in cost_history.items():
        plt.plot(costs, label=house, linewidth=1.5)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title("Training Cost Over Iterations")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # log scale
    plt.subplot(1, 2, 2)
    for house, costs in cost_history.items():
        plt.plot(costs, label=house, linewidth=1.5)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (log scale)')
    plt.title("Training Cost (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # show plots
    plt.tight_layout()
    plt.show()


def save_model(path: str, thetas_dict: dict, houses: list,
              subjects: list, mu: np.ndarray, sigma: np.ndarray) -> None:
    """saves trained model into file"""
    model = {
        "thetas_dict": thetas_dict,
        "houses": houses,
        "subjects": subjects,
        "mu": mu.tolist(),
        "sigma": sigma.tolist()
    }

    with open(path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Model saved to {path}")


def main() -> int:
    """uses gradient descent algo to train a model with logistic regression"""
    parser = argparse.ArgumentParser(
        description="Hogwarts House prediction logistic regression model training"
    )
    parser.add_argument("dataset", help="Path to training CSV (i.e dataset_train.csv)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate (default: 0.5)")
    parser.add_argument("--iters", type=int, default=2000,
                        help="Number of iterations (default: 2000)")
    parser.add_argument("--out", default="model.json",
                        help="Output model file (default: model.json)")
    args = parser.parse_args()

    try:
        # load data
        print(f"Loading data from {args.dataset}")
        data = load(args.dataset)
        if data is None:
            print(f"Error: could not load dataset {args.dataset}")
            return 1

        # extract subjects from data
        subjects = get_numeric_cols(data)
        print(f"{len(subjects)} features found: {subjects}")

        # calculate feature matrix X
        x_raw = data[subjects].values.astype(float)
        print(f"Feature matrix shape: {x_raw.shape}")

        # get house labels and sort
        y_labels = data["Hogwarts House"].values
        houses = sorted(list(set(y_labels)))
        print(f"houses: {houses}")

        # preprocess values
        print(f"Preprocessing...")
        X_fill = replace_nan(x_raw)
        mu, sigma = compute_mu_sig(X_fill)
        X_norm = normalize(X_fill, mu, sigma)
        X_final = add_bias(X_norm)

        # training
        print(f"Training... Params: learning_rate={args.lr}, iterations={args.iters}")
        thetas_dict, cost_hist = train_model(X_final, y_labels, houses, args.lr, args.iters)

        # save model
        print(f"Saving model to {args.out}...")
        save_model(args.out, thetas_dict, houses, subjects, mu, sigma)

        # draw chart
        plot_training_metrics(cost_hist, args.iters)
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
