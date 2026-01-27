import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from load import load
from utils import load_model, sigmoid, preprocess_vals, add_bias, get_numeric_cols


def plot_sigmoid_function():
    """The Sigmoid Function
    Shows how sigmoid squashes any input to [0, 1]

    - Linear regression outputs can be any real number (-∞ to +∞)
    - For classification, we need probabilities (0 to 1)
    - Sigmoid "squashes" the linear output into this range
    - alpha(0) = 0.5 (decision boundary)
    - Large positive z -> alpha(z) ~= 1
    - Large negative z -> alpha(z) ~= 0"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot for sigmoid function
    z = np.linspace(-10, 10, 1000)
    sigma = 1 / (1 + np.exp(-z))

    axes[0].plot(z, sigma, 'b-', linewidth=3, label='alpha(z) = 1/(1+e^-z)')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision boundary (0.5)')
    axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].fill_between(z, 0, sigma, where=(sigma >= 0.5), alpha=0.3, color='green', label='Predict: Class 1')
    axes[0].fill_between(z, 0, sigma, where=(sigma < 0.5), alpha=0.3, color='red', label='Predict: Class 0')
    axes[0].set_xlabel('z = theta^T.x (linear combination)', fontsize=12)
    axes[0].set_ylabel('alpha(z) = Probability', fontsize=12)
    axes[0].set_title('Sigmoid Function: Squashing to Probabilities', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # plot derivative of sigmoid (gradient)
    sigmoid_derivative = sigma * (1 - sigma)
    axes[1].plot(z, sigmoid_derivative, 'purple', linewidth=3, label="alpha'(z) = alpha(z)(1-alpha(z))")
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Max gradient at z=0')
    axes[1].set_xlabel('z = theta^T.x', fontsize=12)
    axes[1].set_ylabel("alpha'(z) = Gradient magnitude", fontsize=12)
    axes[1].set_title('Sigmoid Derivative: Why Vanishing Gradients Happen', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate('Gradient vanishes\nfor extreme values', xy=(7, 0.01), fontsize=10, ha='center', color='red')

    plt.tight_layout()
    plt.show()


def plot_cost_function():
    """Shows why we use log loss instead of MSE

    - MSE with sigmoid creates non-convex surface (many local minima)
    - Log loss is convex (guaranteed global minimum)
    - Heavily penalizes confident wrong predictions
    - Cost = -[y·log(h) + (1-y)·log(1-h)]"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    h = np.linspace(0.001, 0.999, 1000)

    # when y = 1 (actual positive)
    cost_y1 = -np.log(h)
    axes[0].plot(h, cost_y1, 'b-', linewidth=3, label='Cost when y=1: -log(h)')
    axes[0].set_xlabel('Predicted probability h', fontsize=12)
    axes[0].set_ylabel('Cost', fontsize=12)
    axes[0].set_title('Cost When Actual Label y = 1\n(Student IS in this house)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 5)
    axes[0].annotate('Predict 0.99\n-> Cost ~= 0.01\n(Correct!)', xy=(0.95, 0.2), fontsize=10, 
                    ha='center', color='green', fontweight='bold')
    axes[0].annotate('Predict 0.01\n-> Cost ~= 4.6\n(WRONG!)', xy=(0.15, 3.5), fontsize=10,
                    ha='center', color='red', fontweight='bold')

    # when y = 0 (actual negative)
    cost_y0 = -np.log(1 - h)
    axes[1].plot(h, cost_y0, 'r-', linewidth=3, label='Cost when y=0: -log(1-h)')
    axes[1].set_xlabel('Predicted probability h', fontsize=12)
    axes[1].set_ylabel('Cost', fontsize=12)
    axes[1].set_title('Cost When Actual Label y = 0\n(Student is NOT in this house)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 5)
    axes[1].annotate('Predict 0.01\n-> Cost ~= 0.01\n(Correct!)', xy=(0.05, 0.2), fontsize=10,
                    ha='center', color='green', fontweight='bold')
    axes[1].annotate('Predict 0.99\n-> Cost ~= 4.6\n(WRONG!)', xy=(0.85, 3.5), fontsize=10,
                    ha='center', color='red', fontweight='bold')

    plt.tight_layout()

    plt.show()


def plot_gradient_descent_intuition():
    """Shows how gradient descent finds the minimum

    - Start with random theta
    - Compute gradient (slope of cost function)
    - Move opposite to gradient direction
    - Repeat until convergence
    - Learning rate controls step size"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Simple 1D cost function visualization
    theta = np.linspace(-3, 3, 100)
    cost = theta**2 + 0.5  # Simple convex function

    axes[0].plot(theta, cost, 'b-', linewidth=3)

    # simulate gradient descent steps
    theta_path = [2.5]
    lr = 0.3
    for _ in range(8):
        gradient = 2 * theta_path[-1]  # derivative of theta^2
        new_theta = theta_path[-1] - lr * gradient
        theta_path.append(new_theta)

    for i, t in enumerate(theta_path):
        c = t**2 + 0.5
        color = plt.cm.Reds(i / len(theta_path))
        axes[0].scatter(t, c, s=100, c=[color], zorder=5, edgecolors='black')
        if i < len(theta_path) - 1:
            next_t = theta_path[i+1]
            next_c = next_t**2 + 0.5
            axes[0].annotate('', xy=(next_t, next_c), xytext=(t, c),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    axes[0].set_xlabel('theta (weight)', fontsize=12)
    axes[0].set_ylabel('J(theta) (cost)', fontsize=12)
    axes[0].set_title('Gradient Descent: Following the Slope Down', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('Start here', xy=(2.5, 6.75), fontsize=10, ha='center')
    axes[0].annotate('Minimum!', xy=(0, 0.5), fontsize=10, ha='center', color='green', fontweight='bold')

    # learning rate comparison
    lrs = [0.01, 0.1, 0.5, 1.5]
    colors = ['blue', 'green', 'orange', 'red']

    for lr, color in zip(lrs, colors):
        theta_path = [2.5]
        for _ in range(20):
            gradient = 2 * theta_path[-1]
            new_theta = theta_path[-1] - lr * gradient
            theta_path.append(new_theta)
        costs = [t**2 + 0.5 for t in theta_path]
        axes[1].plot(costs, color=color, linewidth=2, label=f'lr={lr}')

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Cost', fontsize=12)
    axes[1].set_title('Effect of Learning Rate', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 10)
    axes[1].annotate('Too small:\nSlow convergence', xy=(15, 3), fontsize=9, color='blue')
    axes[1].annotate('Too large:\nDiverges!', xy=(10, 8), fontsize=9, color='red')

    plt.tight_layout()
    plt.show()


def plot_one_vs_all_strategy():
    """Shows how we convert 4-class to 4 binary problems

    - Logistic regression is inherently binary (0 or 1)
    - For K classes, train K separate classifiers
    - Each asks: "Is this class X or not?"
    - Prediction: pick class with highest probability"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    colors = ['#740001', '#FFD800', '#0E1A40', '#1A472A']

    # Sample data distribution (fake for illustration)
    np.random.seed(42)
    n_samples = 100

    for idx, (ax, house, color) in enumerate(zip(axes.flat, houses, colors)):
        # This house vs all others
        this_house = np.random.randn(n_samples, 2) + np.array([idx % 2 * 3, idx // 2 * 3])
        other_houses = np.random.randn(n_samples * 3, 2) + np.array([1.5, 1.5])

        ax.scatter(other_houses[:, 0], other_houses[:, 1], c='gray', alpha=0.3, 
                s=30, label='Other houses (y=0)')
        ax.scatter(this_house[:, 0], this_house[:, 1], c=color, alpha=0.8, 
                s=50, label=f'{house} (y=1)', edgecolors='black')

        ax.set_title(f'Classifier {idx+1}: {house} vs Rest', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.suptitle('One-vs-All: 4 Binary Classifiers', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_importance():
    """Shows which features matter most for each house

    - theta values show feature importance
    - Large positive theta -> feature increases probability
    - Large negative theta -> feature decreases probability
    - theta₀ (bias) is the baseline probability"""
    try:
        model = load_model('model.json')
    except:
        print("model.json not found. Run logreg_train.py first!")
        return

    thetas_dict = model['thetas_dict']
    subjects = ['Bias'] + model['subjects']
    houses = model['houses']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#740001', '#FFD800', '#0E1A40', '#1A472A']

    for idx, (ax, house, color) in enumerate(zip(axes.flat, houses, colors)):
        thetas = thetas_dict[house]

        # sort by absolute value
        sorted_indices = np.argsort(np.abs(thetas))[::-1]
        sorted_thetas = [thetas[i] for i in sorted_indices]
        sorted_subjects = [subjects[i] for i in sorted_indices]

        bar_colors = ['green' if t > 0 else 'red' for t in sorted_thetas]
        bars = ax.barh(range(len(sorted_thetas)), sorted_thetas, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(sorted_thetas)))
        ax.set_yticklabels(sorted_subjects, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('theta value (weight)', fontsize=10)
        ax.set_title(f'{house}: Feature Importance', fontsize=12, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    fig.text(0.5, 0.02, 'Green = Increases probability | Red = Decreases probability', 
            ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_confusion_matrix():
    """Shows model performance per class

    - Diagonal = correct predictions
    - Off-diagonal = errors (which classes get confused)
    - Useful for identifying model weaknesses"""
    try:
        model = load_model('model.json')
        data = load('datasets/dataset_train.csv')
    except:
        print("model.json or dataset not found!")
        return

    thetas_dict = model['thetas_dict']
    houses = model['houses']
    subjects = model['subjects']
    mu = model['mu']
    sigma = model['sigma']

    # get predictions
    y_true = data['Hogwarts House'].values
    vals = preprocess_vals(data, subjects, mu, sigma)
    biased_vals = add_bias(vals)

    probas = []
    for house in houses:
        theta = np.array(thetas_dict[house], dtype=float)
        proba = sigmoid(biased_vals @ theta)
        probas.append(proba)

    pstack = np.vstack(probas).T
    indices = np.argmax(pstack, axis=1)
    y_pred = [houses[int(i)] for i in indices]

    # build confusion matrix
    n_classes = len(houses)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        i = houses.index(true)
        j = houses.index(pred)
        cm[i, j] += 1

    # calculate accuracy and plot
    accuracy = np.trace(cm) / np.sum(cm) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # confusion matrix heatmap
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].figure.colorbar(im, ax=axes[0])
    axes[0].set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=houses, yticklabels=houses)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')

    axes[0].set_xlabel('Predicted House', fontsize=12)
    axes[0].set_ylabel('True House', fontsize=12)
    axes[0].set_title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)', fontsize=14, fontweight='bold')

    # per class metrics
    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    x = np.arange(n_classes)
    width = 0.25

    axes[1].bar(x - width, precisions, width, label='Precision', color='steelblue')
    axes[1].bar(x, recalls, width, label='Recall', color='darkorange')
    axes[1].bar(x + width, f1s, width, label='F1-Score', color='green')

    axes[1].set_xlabel('House', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(houses, rotation=15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1.1)

    # add values on bars
    for i in range(n_classes):
        axes[1].text(i - width, precisions[i] + 0.02, f'{precisions[i]:.2f}', ha='center', fontsize=8)
        axes[1].text(i, recalls[i] + 0.02, f'{recalls[i]:.2f}', ha='center', fontsize=8)
        axes[1].text(i + width, f1s[i] + 0.02, f'{f1s[i]:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_probability_distribution():
    """Shows confidence of predictions

    - High confidence = probability close to 1
    - Low confidence = probability close to 0.5
    - Overlapping distributions = model uncertainty"""
    try:
        model = load_model('model.json')
        data = load('datasets/dataset_train.csv')
    except:
        print("model.json or dataset not found!")
        return

    thetas_dict = model['thetas_dict']
    houses = model['houses']
    subjects = model['subjects']
    mu = model['mu']
    sigma = model['sigma']

    vals = preprocess_vals(data, subjects, mu, sigma)
    biased_vals = add_bias(vals)
    y_true = data['Hogwarts House'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#740001', '#FFD800', '#0E1A40', '#1A472A']

    for idx, (ax, house, color) in enumerate(zip(axes.flat, houses, colors)):
        theta = np.array(thetas_dict[house], dtype=float)
        proba = sigmoid(biased_vals @ theta).flatten()

        # Split by actual class
        positive_mask = y_true == house
        pos_proba = proba[positive_mask]
        neg_proba = proba[~positive_mask]

        ax.hist(neg_proba, bins=50, alpha=0.5, color='gray', label=f'Not {house}', density=True)
        ax.hist(pos_proba, bins=50, alpha=0.7, color=color, label=f'{house}', density=True)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision boundary')
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{house} Classifier: Probability Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_normalization_effect():
    """Shows why we normalize features

    - Features have different scales
    - Without normalization: large features dominate
    - Z-score: (x - μ) / alpha makes all features comparable
    - After normalization: mean=0, std=1"""
    try:
        data = load('datasets/dataset_train.csv')
    except:
        print("dataset not found!")
        return

    subjects = get_numeric_cols(data)[:6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # before normalization - box plot
    raw_data = [data[s].dropna().values for s in subjects]
    bp = axes[0, 0].boxplot(raw_data, labels=[s[:10] + '...' if len(s) > 10 else s for s in subjects])
    axes[0, 0].set_title('Before Normalization: Different Scales', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # after normalization - box plot
    normalized_data = []
    for s in subjects:
        vals = data[s].dropna().values
        normalized = (vals - np.mean(vals)) / np.std(vals)
        normalized_data.append(normalized)

    axes[0, 1].boxplot(normalized_data, labels=[s[:10] + '...' if len(s) > 10 else s for s in subjects])
    axes[0, 1].set_title('After Z-Score Normalization: Same Scale', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Normalized Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # before normalization - histogram overlay
    for s in subjects[:3]:
        vals = data[s].dropna().values
        axes[1, 0].hist(vals, bins=30, alpha=0.5, label=s[:15])
    axes[1, 0].set_title('Before: Distributions at Different Locations', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # after normalization - histogram overlay
    for s in subjects[:3]:
        vals = data[s].dropna().values
        normalized = (vals - np.mean(vals)) / np.std(vals)
        axes[1, 1].hist(normalized, bins=30, alpha=0.5, label=s[:15])
    axes[1, 1].set_title('After: All Centered at 0', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Normalized Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_decision_boundaries_2d():
    """Shows how the model separates classes


    - Each classifier draws a linear boundary
    - Probability increases/decreases away from boundary
    - One-vs-all: regions where each classifier wins"""
    try:
        model = load_model('model.json')
        data = load('datasets/dataset_train.csv')
    except:
        print("model.json or dataset not found!")
        return

    thetas_dict = model['thetas_dict']
    houses = model['houses']
    subjects = model['subjects']
    mu = np.array(model['mu'])
    sigma = np.array(model['sigma'])

    # use 2 most important features by variance of thetas across houses
    all_thetas = np.array([thetas_dict[h] for h in houses])
    theta_variance = np.var(all_thetas[:, 1:], axis=0)
    top_2_idx = np.argsort(theta_variance)[-2:]

    feat1_name = subjects[top_2_idx[0]]
    feat2_name = subjects[top_2_idx[1]]

    # get raw data and normalize
    f1_raw = data[feat1_name].fillna(data[feat1_name].mean()).values
    f2_raw = data[feat2_name].fillna(data[feat2_name].mean()).values

    f1 = (f1_raw - mu[top_2_idx[0]]) / sigma[top_2_idx[0]]
    f2 = (f2_raw - mu[top_2_idx[1]]) / sigma[top_2_idx[1]]

    y_true = data['Hogwarts House'].values

    # make grid
    x_min, x_max = f1.min() - 0.5, f1.max() + 0.5
    y_min, y_max = f2.min() - 0.5, f2.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))

    # build simplified thetas
    grid_features = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]

    predictions = np.zeros(xx.ravel().shape, dtype=int)
    max_proba = np.zeros(xx.ravel().shape)

    for h_idx, house in enumerate(houses):
        full_theta = np.array(thetas_dict[house])
        # use only bias and the 2 selected features
        simple_theta = np.array([full_theta[0], 
                                full_theta[top_2_idx[0] + 1], 
                                full_theta[top_2_idx[1] + 1]])
        proba = sigmoid(grid_features @ simple_theta)

        mask = proba > max_proba
        predictions[mask] = h_idx
        max_proba[mask] = proba[mask]

    predictions = predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors_map = ListedColormap(['#740001', '#FFD800', '#0E1A40', '#1A472A'])
    ax.contourf(xx, yy, predictions, alpha=0.3, cmap=colors_map)
    ax.contour(xx, yy, predictions, colors='black', linewidths=0.5)

    house_colors = {'Gryffindor': '#740001', 'Hufflepuff': '#FFD800', 
                    'Ravenclaw': '#0E1A40', 'Slytherin': '#1A472A'}

    for house in houses:
        mask = y_true == house
        ax.scatter(f1[mask], f2[mask], c=house_colors[house], label=house,
                edgecolors='white', s=40, alpha=0.7)

    ax.set_xlabel(f'{feat1_name} (normalized)', fontsize=12)
    ax.set_ylabel(f'{feat2_name} (normalized)', fontsize=12)
    ax.set_title('Decision Boundaries (2D Projection)\nColored regions show classifier predictions', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """generates visualizations for in depth explanation of functions used"""
    print("\nVisualization 1: Sigmoid Function")
    plot_sigmoid_function()

    print("\nVisualization 2: Log Loss Cost Function")
    plot_cost_function()

    print("\nVisualization 3: Gradient Descent Intuition")
    plot_gradient_descent_intuition()

    print("\nVisualization 4: One-vs-All  ")
    plot_one_vs_all_strategy()

    print("\nVisualization 5: Feature Importance")
    plot_feature_importance()

    print("\nVisualization 6: Confusion Matrix & Metrics")
    plot_confusion_matrix()

    print("\nVisualization 7: Probability Distributions")
    plot_probability_distribution()

    print("\nVisualization 8: Normalization Effect")
    plot_normalization_effect()

    print("\nVisualization 9: Decision Boundaries")
    plot_decision_boundaries_2d()


if __name__ == "__main__":
    main()
