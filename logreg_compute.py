import math
import matplotlib.pyplot as plt
from typing import List, Dict
from logreg_train import get_km_price
from logreg_predict import estimate_price, normalize_data, load_model


def compute(price_list: List[float], preds: List[float]) -> Dict[str, float]:
    """computes metrics using list of prices and predictions"""
    sample_count = len(price_list)

    #  mean absolute error
    mae = 0.0
    for actual, predicted in zip(price_list, preds):
        mae += abs(actual - predicted)
    mae /= sample_count

    # mean square error
    mse = 0.0
    for actual, predicted in zip(price_list, preds):
        mse += (actual - predicted) ** 2
    mse /= sample_count

    # root mean squared error
    rmse = math.sqrt(mse)

    # coefficient of determination
    mean_price = sum(price_list) / sample_count

    # total sum of squares (variance)
    ss_total = 0.0
    for price in price_list:
        ss_total += (price - mean_price) ** 2

    # residual sum of squares (variance prediction error)
    ss_residual = 0.0
    for actual, predicted in zip(price_list, preds):
        ss_residual += (actual - predicted) ** 2

    # r2 squared coefficient of determination
    if ss_total != 0:
        r2 = 1.0 - (ss_residual / ss_total)
    else:
        r2 = 0.0

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def pred_from_raw(x: float, normalized: bool, x_min: float, x_max: float,
                  theta0: float, theta1: float) -> float:
    x_used = x
    if normalized:
        x_used = normalize_data(x, float(x_min), float(x_max))
    return estimate_price(x_used, theta0, theta1)


def draw_chart(km_list_raw: list, price_list: list,
               y1: float, y2: float, preds: list, metrics: dict):
    """draws plots for linear regression and metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # linear regression fit
    axes[0, 0].scatter(km_list_raw, price_list, label="Actual Data",
                       alpha=0.75, s=60, edgecolors='black')
    axes[0, 0].plot([min(km_list_raw), max(km_list_raw)], [y1, y2], 'r-',
                    linewidth=2, label="Regression Line")
    axes[0, 0].set_title("Linear Regression", fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Mileage (km)", fontsize=12)
    axes[0, 0].set_ylabel("Price", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # actual vs predicted price
    axes[0, 1].scatter(price_list, preds, alpha=0.75, s=60, edgecolors='black')
    min_val = min(min(price_list), min(preds))
    max_val = max(max(price_list), max(preds))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--',
                    linewidth=2, label="Perfect Prediction")
    axes[0, 1].set_title("Actual vs Predicted Prices", fontsize=14,
                         fontweight='bold')
    axes[0, 1].set_xlabel("Actual Price", fontsize=12)
    axes[0, 1].set_ylabel("Predicted Price", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # residuals
    residuals = [actual - pred for actual, pred in zip(price_list, preds)]
    axes[1, 0].scatter(km_list_raw, residuals, alpha=0.75, s=60,
                       edgecolors='black')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title("Residuals Plot", fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel("Mileage (km)", fontsize=12)
    axes[1, 0].set_ylabel("Residual", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # metrics summary
    axes[1, 1].axis('off')
    metrics_text = "Model Performance Metrics\n\n"
    metrics_text += f"Rsquared Score: {metrics['R2']:.4f}\n"
    metrics_text += "(Closer to 1 = better fit)\n\n"
    metrics_text += f"MAE: -+{metrics['MAE']:.2f}\n(Mean Absolute Error)\n\n"
    metrics_text += f"RMSE: -+{metrics['RMSE']:.2f}\n"
    metrics_text += "(Root Mean Squared Error)\n\n"
    metrics_text += f"Data Points: {len(price_list)}\n"
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=13, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round'))

    plt.tight_layout()
    plt.show()


def predict_subjects(col: np.ndarray, subjects: List[str], thetas: Dict[str, List[float]]) -> List[str]:
    probas = []
    for house in subjects:
        theta = np.array(thetas[house], dtype=float)
        proba = sigmoid(col @ theta)
        probas.append(proba)

    pstack = np.vstack(probas).T
    indices = np.argmax(pstack, axis=1)
    return [subjects[int(i)] for i in indices]




def main() -> int:
    """plots the linear regression and calculates precision metrics."""
    # load trained model
    model = load_model("model.json")
    theta0 = float(model["theta0"])
    theta1 = float(model["theta1"])
    normalized = bool(model.get("normalized", False))
    x_min = model.get("x_min", None)
    x_max = model.get("x_max", None)

    km_list_raw, price_list = get_km_price("data.csv")

    preds = []
    for km in km_list_raw:
        x_used = km
        if normalized:
            x_used = normalize_data(km, float(x_min), float(x_max))
        preds.append(estimate_price(x_used, theta0, theta1))

    metrics = compute(price_list, preds)
    print("metrics from training data:")
    for key, val in metrics.items():
        print(f"{key}: {val:.6f}")

    y1 = pred_from_raw(min(km_list_raw), normalized,
                       x_min, x_max, theta0, theta1)
    y2 = pred_from_raw(max(km_list_raw), normalized,
                       x_min, x_max, theta0, theta1)
    draw_chart(km_list_raw, price_list, y1, y2, preds, metrics)

    return 0


if __name__ == "__main__":
    main()
