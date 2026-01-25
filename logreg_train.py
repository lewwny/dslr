import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from load import load
from utils import get_numeric_cols, load_model, sigmoid, arr_tofloat


def normalize_data(km_list: List[float]) -> Tuple[List[float], float, float]:
    """data normalizer using the list of km values"""
    x_min = min(km_list)
    x_max = max(km_list)
    if x_min == x_max:
        raise ValueError("Cannot train based on values given")
    km_normalized = []
    for km in km_list:
        km_normalized.append((km - x_min) / (x_max - x_min))
    return km_normalized, x_min, x_max


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """price estimation using the function from the subject"""
    return theta0 + (theta1 * mileage)


def gradient_descent(theta0: float, theta1: float, learning_rate: float,
                     iterations: int, mileage_list: List[float],
                     price_list: List[float]
                     ) -> Tuple[float, float, List[float]]:
    """Gradient descent algorithm for linear regression
    tmp_t0 = l_rate*(1/m)*sum(est_price(mileage[i])-price[i])
    tmp_t1 = l_rate*(1/m)*sum((est_price(mileage[i])-price[i])*mileage[i])"""
    val_count = len(mileage_list)
    loss_history = []

    for i in range(iterations):
        # sum of errors for theta0 and theta1
        sum_error_theta0 = 0.0
        sum_error_theta1 = 0.0

        for j in range(val_count):
            prediction = estimate_price(mileage_list[j], theta0, theta1)
            error = prediction - price_list[j]
            sum_error_theta0 += error
            sum_error_theta1 += error * mileage_list[j]

        # temp tethas
        tmp_theta0 = learning_rate * (1 / val_count) * sum_error_theta0
        tmp_theta1 = learning_rate * (1 / val_count) * sum_error_theta1

        # update thetas
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

        # MSE for this iteration
        mse = 0.0
        for j in range(val_count):
            prediction = estimate_price(mileage_list[j], theta0, theta1)
            mse += (prediction - price_list[j]) ** 2
        mse /= val_count
        loss_history.append(mse)

        if (i + 1) % 100 == 0:
            itermessage = f"Iteration {i + 1}: theta0 = {theta0:.6f},"
            itermessage += f"theta1 = {theta1:.6f}, MSE = {mse:.2f}"
            print(itermessage)

    return theta0, theta1, loss_history


def plot_training_metrics(loss_history: List[float], iterations: int) -> None:
    """Visualize training progress: loss over iterations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # loss over all iterations
    axes[0].plot(range(iterations), loss_history, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Training Loss Over Iterations', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # log scale to see convergence
    axes[1].plot(range(iterations), loss_history, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('MSE Loss (log scale)', fontsize=12)
    axes[1].set_title('Training Loss (Log Scale)', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_file(path: str, model: Dict[str, any]) -> None:
    """saves trained model into file"""
    with open(path, 'w') as f:
        json.dump(model, f, indent=2)


def get_km_price(file: str) -> Tuple[List[float], List[float]]:
    """gets values for km and price from csv, returns as tuple of lists"""
    data = load(file)
    if data is None:
        raise FileNotFoundError("Could not open file")
    if "km" not in data.columns or "price" not in data.columns:
        raise ValueError("csv doesn't contain the needed values")

    km_list = [float(val) for val in data["km"].tolist()]
    price_list = [float(val) for val in data["price"].tolist()]

    if len(km_list) < 2:
        raise ValueError("csv must contain at least two rows")

    return km_list, price_list


def main() -> int:
    """uses gradient descent algo to train a model to predict price"""
    start = 0
    learn = 0.1
    iters = 1500
    use_norm = True

    try:
        km_list_raw, price_list = get_km_price("data.csv")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    model: Dict[str, any] = {}
    if use_norm:
        try:
            km_list, x_min, x_max = normalize_data(km_list_raw)
        except Exception as e:
            print(f"Error: {e}")
            return 1
        model["normalized"] = True
        model["x_min"] = x_min
        model["x_max"] = x_max
    else:
        km_list = km_list_raw
        model["normalized"] = False
        model["x_min"] = None
        model["x_max"] = None
        if learn == 0.1:
            learn = 1e-8

    theta0, theta1, loss_history = gradient_descent(start, start, learn,
                                                    iters, km_list, price_list)
    model["theta0"] = theta0
    model["theta1"] = theta1
    model["start"] = start
    model["learning_rate"] = learn
    model["iterations"] = iters

    output_file = "model.json"
    save_file(output_file, model)
    print(f"Finished training model and saved to {output_file}")
    values = f"Values: theta0={theta0:.6f}, theta1={theta1:.6f},"
    values += f"normalized={model['normalized']}"
    print(values)
    print(f"Final MSE loss: {loss_history[-1]:.2f}")

    plot_training_metrics(loss_history, iters)

    return 0


if __name__ == "__main__":
    main()
