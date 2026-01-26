import numpy as np
from utils import sigmoid
from logreg_train import compute_cost


def sgd(X: np.ndarray, y: np.ndarray, theta: np.ndarray,
        lr: float, epochs: int, verbose: bool = True):
    """Stochastic Gradient Descent (SGD) algo
    Difference between GD and SGD:
    - batch GD (classic): uses all samples to compute gradient (slow but stable)
    - SGD: uses one sample at a time (fast but noisy)

    Advantages:
    -For large datasets, computing gradient is slow, but SGD updates after each sample,
     so learning starts immediately
    - The noise from single-sample gradients helps escape shallow local minima
    - Can learn from streaming data

    Disadvantages:
    - More iterations to converge
    - Noisy cost curve
    - May oscillate near minimum

    epoch: one pass through the entire dataset
    shuffling: to avoid learning patterns from data order we shuffle data each epoch
    """
    m = len(y)
    cost_hist = []

    for epoch in range(epochs):
        # shuffle data each epoch to prevent cycles and falling into local minima
        indices = np.random.permutation(m)
        X_shuff = X[indices]
        y_shuff = y[indices]
        
        for i in range(m):
            # take a random sample
            xi = X_shuff[i:i+1]
            yi = y_shuff[i:i+1]
            
            # pass to sigmoid function
            hi = sigmoid(xi @ theta)
            
            # gradient from single sample
            gradient = xi.T @ (hi - yi)

            # update
            theta = theta - lr * gradient

        # compute on full dataset for monitorinbg
        h_full = sigmoid(X @ theta)
        cost = compute_cost(h_full, y)
        cost_hist.append(cost)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Cost: {cost:.6f}")
    
    return theta, cost_hist


def minibatch(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lr: float,
              epochs: int, batch_size: int = 32, verbose: bool = True):
    """Mini-Batch Gradient Descent.
    - Batch GD: Uses ALL samples -> stable but slow
    - SGD: Uses 1 sample -> fast but noisy
    - Mini-batch: Uses BATCH_SIZE samples -> balanced

    Advantages:
        - vectorization: Modern CPUs/GPUs are optimized for matrix operations, so processing
            32 samples is barely slower than 1 sample
        - stable gradients: taking the average of 32 samples reduces noise vs taking 1 sample,
            while still being faster than batch gd
        - memory efficiency: for huge datasets bigger than memory, mini-batches can be loaded one at a time

    batch sizes (32, 64, 128, 256...) for matrix operations
    - smaller -> more noise, might generalize better
    - larger -> more stable, faster on GPU, might overfit"""
    m = len(y)
    cost_hist = []

    for epoch in range(epochs):
        # shuffle data each epoch
        indices = np.random.permutation(m)
        X_shuff = X[indices]
        y_shuff = y[indices]

        # process in batches using batch_size
        for i in range(0, m, batch_size):
            # extract batch
            X_batch = X_shuff[i:i+batch_size]
            y_batch = y_shuff[i:i+batch_size]

            # forward pass to sigmoid func
            h_batch = sigmoid(X_batch @ theta)

            # gradient from batch avg
            batch_m = len(y_batch)
            gradient = (1 / batch_m) * (X_batch.T @ (h_batch - y_batch))

            # update
            theta = theta - lr * gradient

        # compute cost on full dataset for monitoring
        h_full = sigmoid(X @ theta)
        cost = compute_cost(h_full, y)
        cost_hist.append(cost)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Cost: {cost:.6f}")

    return theta, cost_hist