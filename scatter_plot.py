

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