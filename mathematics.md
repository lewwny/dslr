## Logistic Regression for Hogwarts House Classification

---

# Part 1: Core Mathematical Concepts

## 1.1 What is Logistic Regression?

**Linear Regression** predicts continuous values:
```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `ŷ` | Predicted output (y-hat) | The raw prediction before sigmoid |
| `θ₀` | Bias/intercept term | Baseline probability offset |
| `θ₁, θ₂, ..., θₙ` | Weights for each feature | How much each subject affects the prediction |
| `x₁, x₂, ..., xₙ` | Input features | Student's grades in each subject |
| `n` | Number of features | 13 (number of Hogwarts subjects) |
| `θᵀ` | Theta transpose | Row vector of all weights [θ₀, θ₁, ..., θₙ] |
| `θᵀx` | Dot product | θ₀·1 + θ₁·x₁ + θ₂·x₂ + ... |

**Problem**: This can output any real number (-∞ to +∞), but we need probabilities (0 to 1).

**Solution**: Wrap it in the **sigmoid function**:
```
h(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `h(x)` | Hypothesis function | Predicted probability (0 to 1) |
| `σ()` | Sigma - the sigmoid function | Squashing function |
| `e` | Euler's number ≈ 2.71828 | Mathematical constant |
| `θᵀx` | Linear combination | Sum of weighted features |
| `-θᵀx` | Negated linear combination | Input to exponential |

### Properties of Sigmoid (see viz_1_sigmoid.png):
| Input z = θᵀx | Output σ(z) | Interpretation |
|---------------|-------------|----------------|
| z → +∞ | σ(z) → 1 | High confidence: Class 1 |
| z = 0 | σ(z) = 0.5 | Uncertain (decision boundary) |
| z → -∞ | σ(z) → 0 | High confidence: Class 0 |

**Key Code** (`utils.py`):
```python
def sigmoid(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-arr))
```

---

## 1.2 Why Log Loss Instead of MSE?

**Mean Squared Error (MSE)** works for linear regression:
```
J(θ) = (1/m) Σ(h(xⁱ) - yⁱ)²
```

### Symbol Definitions for MSE:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `J(θ)` | Cost function (J of theta) | Total error of the model |
| `m` | Number of training samples | 1600 students in dataset_train.csv |
| `1/m` | Average | Divide by sample count to get mean |
| `Σ` | Sigma - summation | Sum over all samples i=1 to m |
| `h(xⁱ)` | Prediction for sample i | Predicted probability for student i |
| `xⁱ` | Features of sample i | Grades of student i |
| `yⁱ` | True label for sample i | Actual house of student i (0 or 1) |
| `(...)²` | Squared | Square the error |

**Problem with MSE + Sigmoid**: Creates a non-convex cost surface with many local minima. Gradient descent gets stuck!

**Solution: Log Loss (Binary Cross-Entropy)**:
```
J(θ) = -(1/m) Σᵢ₌₁ᵐ [yⁱ·log(h(xⁱ)) + (1-yⁱ)·log(1-h(xⁱ))]
```

### Symbol Definitions for Log Loss:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `J(θ)` | Cost function | Total error to minimize |
| `-` | Negative sign | Makes cost positive (log of values <1 is negative) |
| `1/m` | Average | Mean over all samples |
| `Σᵢ₌₁ᵐ` | Sum from i=1 to m | Sum over all 1600 students |
| `yⁱ` | True label (0 or 1) | 1 if student i is in this house, 0 otherwise |
| `log()` | Natural logarithm (ln) | Base-e logarithm |
| `h(xⁱ)` | Predicted probability | Model's confidence for student i |
| `1-yⁱ` | Inverse label | 1 if NOT in house, 0 if in house |
| `1-h(xⁱ)` | Inverse probability | Probability of NOT being in house |

### Breaking Down the Formula:

**When yⁱ = 1** (student IS in this house):
```
Cost = -log(h(xⁱ))
```
- If h = 0.99 (correct): -log(0.99) ≈ 0.01 ✓ LOW
- If h = 0.01 (wrong): -log(0.01) ≈ 4.6 ✗ HIGH

**When yⁱ = 0** (student is NOT in this house):
```
Cost = -log(1-h(xⁱ))
```
- If h = 0.01 (correct): -log(0.99) ≈ 0.01 ✓ LOW
- If h = 0.99 (wrong): -log(0.01) ≈ 4.6 ✗ HIGH

**Key Code** (`utils.py`):
```python
def compute_cost(h: np.ndarray, y: np.ndarray) -> float:
    m = len(y)                          # Number of samples
    eps = 1e-15                         # Prevent log(0)
    h_clip = np.clip(h, eps, 1 - eps)   # Clip to avoid log(0)
    
    # The formula: -1/m * sum[y*log(h) + (1-y)*log(1-h)]
    cost = -1/m * np.sum(y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip))
    return cost
```

---

## 1.3 Gradient Descent

**Goal**: Find θ that minimizes J(θ)

**Method**: Iteratively move opposite to the gradient:
```
θ := θ - α · ∇J(θ)
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `θ` | Weight vector | [θ₀, θ₁, ..., θ₁₃] - 14 values (bias + 13 features) |
| `:=` | Assignment | Update θ with new value |
| `α` | Alpha - learning rate | Step size = 0.5 (default) |
| `∇` | Nabla - gradient operator | Vector of partial derivatives |
| `∇J(θ)` | Gradient of cost | Direction of steepest increase |
| `α · ∇J(θ)` | Scaled gradient | How much to move |
| `-α · ∇J(θ)` | Negative gradient | Move OPPOSITE to steepest increase |

### The Gradient Formula (Partial Derivative):

For each weight θⱼ:
```
∂J/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ) · xⱼⁱ
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `∂J/∂θⱼ` | Partial derivative of J w.r.t. θⱼ | How J changes when θⱼ changes |
| `j` | Index of weight | j=0 is bias, j=1 is Arithmancy, etc. |
| `1/m` | Average | Mean over all samples |
| `Σᵢ₌₁ᵐ` | Sum from i=1 to m | Sum over all students |
| `h(xⁱ)` | Predicted probability | What we predicted |
| `yⁱ` | True label | What it actually is |
| `h(xⁱ) - yⁱ` | Error | How wrong we were |
| `xⱼⁱ` | Feature j of sample i | Student i's grade in subject j |

### In Matrix Form (much faster):
```
∇J(θ) = (1/m) · Xᵀ · (h - y)
```

### Symbol Definitions:
| Symbol | Meaning | Dimensions in Our Project |
|--------|---------|---------------------------|
| `∇J(θ)` | Gradient vector | (14, 1) - one gradient per weight |
| `1/m` | Average factor | Scalar (1/1600) |
| `X` | Feature matrix | (1600, 14) - students × features |
| `Xᵀ` | X transpose | (14, 1600) - features × students |
| `h` | Predictions vector | (1600, 1) - one prediction per student |
| `y` | Labels vector | (1600, 1) - one label per student |
| `h - y` | Error vector | (1600, 1) - error per student |
| `Xᵀ · (h-y)` | Matrix multiplication | (14, 1) - aggregated gradient |

**Key Code** (`logreg_train.py`):
```python
def gradient_descent(X, y, theta, learning_rate, iterations, verb=True):
    m = len(y)  # m = number of samples (1600)
    cost_hist = []

    for i in range(iterations):
        # Step 1: Forward pass - compute predictions
        z = X @ theta           # z = Xθ (linear combination)
        h = sigmoid(z)          # h = σ(z) (squash to probability)
        
        # Step 2: Compute gradient using matrix form
        # ∇J(θ) = (1/m) · Xᵀ · (h - y)
        gradient = (1/m) * (X.T @ (h - y))
        
        # Step 3: Update weights
        # θ := θ - α · ∇J(θ)
        theta = theta - learning_rate * gradient
        
        # Track cost
        cost = compute_cost(h, y)
        cost_hist.append(cost)

    return theta, cost_hist
```

### Learning Rate Effects (see viz_3_gradient_descent.png):

| Learning Rate α | Effect |
|-----------------|--------|
| Too small (0.01) | Slow convergence, many iterations needed |
| Good (0.1-0.5) | Smooth convergence |
| Too large (1.5+) | Diverges! Cost explodes |

---

## 1.4 One-vs-All (One-vs-Rest) Strategy

**Problem**: Logistic regression is binary (0 or 1), but we have 4 houses.

**Solution**: Train 4 separate binary classifiers!

### The Strategy:

For K classes, train K classifiers:
```
For k = 1 to K:
    Train classifier hₖ(x) where y = 1 if class k, else y = 0
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `K` | Number of classes | 4 (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) |
| `k` | Class index | 1=Gryffindor, 2=Hufflepuff, 3=Ravenclaw, 4=Slytherin |
| `hₖ(x)` | Classifier for class k | Probability student x is in house k |
| `θₖ` | Weights for class k | 14 weights specific to house k |

| Classifier | Question | y=1 | y=0 |
|------------|----------|-----|-----|
| θ_Gryffindor | "Is this Gryffindor?" | 327 Gryffindor students | 1273 others |
| θ_Hufflepuff | "Is this Hufflepuff?" | 529 Hufflepuff students | 1071 others |
| θ_Ravenclaw | "Is this Ravenclaw?" | 443 Ravenclaw students | 1157 others |
| θ_Slytherin | "Is this Slytherin?" | 301 Slytherin students | 1299 others |

### Prediction Formula:
```
ŷ = argmax(h₁(x), h₂(x), h₃(x), h₄(x))
     k∈{1,2,3,4}
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `ŷ` | Predicted class | The house we assign the student to |
| `argmax` | Argument of maximum | Return the k that gives highest value |
| `k∈{1,2,3,4}` | k in set of classes | Over all 4 houses |
| `hₖ(x)` | Probability of class k | P(student is in house k) |

**Example**:
```
Student X:
  h₁(x) = P(Gryffindor) = 0.85  ← WINNER
  h₂(x) = P(Hufflepuff) = 0.12
  h₃(x) = P(Ravenclaw)  = 0.03
  h₄(x) = P(Slytherin)  = 0.02
  
  ŷ = argmax = Gryffindor
```

**Key Code** (`logreg_train.py`):
```python
def train_model(X, y_labels, houses, lr, iters, method):
    thetas_dict = {}
    
    for house in houses:  # For each of K=4 classes
        # Create binary labels: yⁱ = 1 if student i in this house, else 0
        y_bin = (y_labels == house).astype(float).reshape(-1, 1)
        
        # Initialize θₖ = zeros
        theta = np.zeros((X.shape[1], 1))  # Shape: (14, 1)
        
        # Train classifier hₖ
        theta, cost_hist = gradient_descent(X, y_bin, theta, lr, iters)
        
        thetas_dict[house] = theta.flatten().tolist()
    
    return thetas_dict  # Returns {house: θₖ} for each house
```

**Key Code** (`logreg_predict.py`):
```python
probas = []
for house in houses:  # For k = 1 to K
    theta = np.array(thetas_dict[house])  # θₖ
    proba = sigmoid(biased_vals @ theta)   # hₖ(x) = σ(θₖᵀx)
    probas.append(proba)

# Stack: each row is a student, each column is P(house)
pstack = np.vstack(probas).T  # Shape: (n_samples, 4)

# argmax over columns (houses)
indices = np.argmax(pstack, axis=1)  # Index of max probability

# Map index to house name
preds = [houses[i] for i in indices]  # ŷ for each student
```

---

# Part 2: Data Preprocessing

## 2.1 Handling Missing Values (NaN)

**Problem**: ML algorithms can't handle NaN values.

**Solution**: Replace NaN with column mean:
```
x_missing = μⱼ (mean of column j)
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `NaN` | Not a Number | Missing grade value |
| `μⱼ` | Mean of feature j | Average grade in subject j |
| `x_missing` | Imputed value | Replaced missing value |

```python
def replace_nan(arr: np.ndarray) -> np.ndarray:
    copy = arr.copy()
    for col_idx in range(copy.shape[1]):  # For each feature j
        col = copy[:, col_idx]
        valid_mask = ~np.isnan(col)        # Non-NaN values
        col_mean = np.mean(col[valid_mask]) # μⱼ = mean of valid values
        copy[np.isnan(col), col_idx] = col_mean  # Replace NaN with μⱼ
    return copy
```

---

## 2.2 Feature Normalization (Z-Score)

**Problem**: Features have vastly different scales:
- Arithmancy: -24,000 to 105,000
- Herbology: -10 to 10

**Solution: Z-Score Normalization**:
```
x_norm = (x - μ) / σ
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `x` | Original feature value | Raw grade |
| `x_norm` | Normalized value | Standardized grade |
| `μ` | Mu - population mean | Average grade in that subject |
| `σ` | Sigma - standard deviation | Spread of grades |
| `x - μ` | Centered value | Distance from mean |
| `(x - μ)/σ` | Standardized value | Distance in units of std |

### After normalization:
- All features: μ ≈ 0, σ ≈ 1
- All features on same scale

```python
def compute_mu_sig(arr):
    mu = np.mean(arr, axis=0)     # μⱼ for each feature j
    sigma = np.std(arr, axis=0)   # σⱼ for each feature j
    sigma[sigma == 0] = 1.0       # Avoid division by zero
    return mu, sigma

def normalize(arr, mu, sigma):
    return (arr - mu) / sigma     # x_norm = (x - μ) / σ
```

**CRITICAL**: Save μ and σ during training! Test data must be normalized the same way.

---

## 2.3 Adding the Bias Term

**Problem**: Our equation needs an intercept:
```
z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

**Solution**: Add x₀ = 1 so we can use matrix multiplication:
```
z = θ₀·x₀ + θ₁x₁ + ... = θᵀx where x₀ = 1
```

### Symbol Definitions:
| Symbol | Meaning | In Our Project |
|--------|---------|----------------|
| `x₀` | Bias feature | Always = 1 |
| `θ₀` | Bias weight | Baseline log-odds |
| `[1, x₁, x₂, ...]` | Augmented feature vector | 14 values (1 + 13 subjects) |

```python
def add_bias(arr: np.ndarray) -> np.ndarray:
    ones = np.ones((arr.shape[0], 1))  # Column of 1s
    return np.concatenate([ones, arr], axis=1)  # Prepend to features
```

**Before**: X shape = (1600, 13)
**After**: X shape = (1600, 14) with first column all 1s

---

# Part 3: Code Walkthrough

## 3.1 Training Flow (`logreg_train.py`)

```
1. LOAD DATA
   └── Load dataset_train.csv → m = 1600 samples, n = 13 features

2. PREPROCESS
   └── Replace NaN with μⱼ (column means)
   └── Compute μ, σ for each feature
   └── Normalize: x_norm = (x - μ) / σ
   └── Add bias: X becomes (m, n+1) = (1600, 14)

3. TRAIN (One-vs-All) for K = 4 classes
   └── For each house k:
       ├── Create yₖ: 1 if this house, 0 otherwise
       ├── Initialize θₖ = [0, 0, ..., 0] (14 zeros)
       ├── For i = 1 to iterations:
       │   ├── h = σ(X·θₖ)              Forward pass
       │   ├── ∇J = (1/m)·Xᵀ·(h-yₖ)    Compute gradient
       │   └── θₖ = θₖ - α·∇J           Update weights
       └── Save θₖ

4. SAVE MODEL (model.json)
   ├── thetas_dict: {house: [θ₀, θ₁, ..., θ₁₃]}
   ├── houses: ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
   ├── subjects: ["Arithmancy", "Astronomy", ...]
   ├── mu: [μ₁, μ₂, ..., μ₁₃]
   └── sigma: [σ₁, σ₂, ..., σ₁₃]
```

## 3.2 Prediction Flow (`logreg_predict.py`)

```
1. LOAD MODEL
   └── Load θₖ for each house, plus μ, σ

2. LOAD TEST DATA
   └── Load dataset_test.csv → m_test samples

3. PREPROCESS (identical to training!)
   └── For NaN: replace with μⱼ (from training!)
   └── Normalize: x_norm = (x - μ) / σ (using training μ, σ!)
   └── Add bias column

4. PREDICT
   └── For each house k:
   │   └── hₖ(x) = σ(X·θₖ)  Compute probability
   └── ŷ = argmax(h₁, h₂, h₃, h₄)  Pick highest

5. OUTPUT
   └── Write houses.csv with predictions
```

---

# Part 4: Bonus Implementations

## 4.1 Stochastic Gradient Descent (SGD)

**Batch GD**: Uses all m samples per update
**SGD**: Uses 1 sample per update

### Update Rule:
```
For each sample i:
    θ := θ - α · ∇Jᵢ(θ)
```

### Symbol Definitions:
| Symbol | Meaning | Difference from Batch GD |
|--------|---------|--------------------------|
| `∇Jᵢ(θ)` | Gradient from sample i only | Not averaged over all m |
| `α` | Learning rate | Usually smaller for SGD |
| `epoch` | One pass through all data | See all m samples once |

```python
def sgd(X, y, theta, lr, epochs):
    m = len(y)
    for epoch in range(epochs):
        # Shuffle to avoid patterns
        indices = np.random.permutation(m)
        X_shuff, y_shuff = X[indices], y[indices]
        
        for i in range(m):  # One sample at a time
            xi = X_shuff[i:i+1]           # Shape: (1, 14)
            yi = y_shuff[i:i+1]           # Shape: (1, 1)
            hi = sigmoid(xi @ theta)      # hᵢ = σ(xᵢᵀθ)
            gradient = xi.T @ (hi - yi)   # No 1/m because m=1
            theta = theta - lr * gradient # Update after each sample
    
    return theta
```

---

## 4.2 Mini-Batch Gradient Descent

**Best of both worlds**: Use B samples per update (B = batch size)

### Update Rule:
```
For each batch b of size B:
    θ := θ - α · (1/B) · Σᵢ∈ᵦ ∇Jᵢ(θ)
```

### Symbol Definitions:
| Symbol | Meaning | Typical Values |
|--------|---------|----------------|
| `B` | Batch size | 32, 64, 128, 256 |
| `b` | Current batch | Subset of training data |
| `Σᵢ∈ᵦ` | Sum over samples in batch | Average B gradients |
| `1/B` | Batch average | Normalize by batch size |

```python
def minibatch(X, y, theta, lr, epochs, batch_size=32):
    m = len(y)
    B = batch_size  # Typically 32
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuff, y_shuff = X[indices], y[indices]
        
        for i in range(0, m, B):  # Step by batch size
            X_batch = X_shuff[i:i+B]      # B samples
            y_batch = y_shuff[i:i+B]      # B labels
            
            h_batch = sigmoid(X_batch @ theta)
            # Gradient averaged over batch
            gradient = (1/len(y_batch)) * (X_batch.T @ (h_batch - y_batch))
            theta = theta - lr * gradient
    
    return theta
```

### Comparison:
| Method | Samples per Update | Memory | Stability | Speed |
|--------|-------------------|--------|-----------|-------|
| Batch GD | m (all) | High | Very stable | Slow |
| SGD | 1 | Low | Noisy | Fast per update |
| Mini-batch | B (32-256) | Medium | Balanced | Best overall |

---

# Part 5: Key Formulas Summary

| Concept | Formula | Symbol Meanings |
|---------|---------|-----------------|
| **Sigmoid** | σ(z) = 1/(1+e⁻ᶻ) | z=input, e=2.718... |
| **Hypothesis** | h(x) = σ(θᵀx) | θ=weights, x=features |
| **Log Loss** | J = -(1/m)Σ[y·log(h)+(1-y)·log(1-h)] | m=samples, y=label, h=prediction |
| **Gradient** | ∇J = (1/m)·Xᵀ(h-y) | X=features matrix, h-y=errors |
| **Update** | θ := θ - α·∇J | α=learning rate |
| **Z-Score** | x_norm = (x-μ)/σ | μ=mean, σ=std |
| **Prediction** | ŷ = argmax(h₁,...,hₖ) | Pick class with max probability |

---

# Part 6: Dimensions Reference

| Variable | Shape | Meaning |
|----------|-------|---------|
| `X` | (m, n+1) = (1600, 14) | m samples, n features + bias |
| `y` | (m, 1) = (1600, 1) | m binary labels |
| `θ` | (n+1, 1) = (14, 1) | n+1 weights (incl. bias) |
| `h` | (m, 1) = (1600, 1) | m predictions |
| `∇J` | (n+1, 1) = (14, 1) | Gradient per weight |
| `μ` | (n,) = (13,) | Mean per feature |
| `σ` | (n,) = (13,) | Std per feature |

---

# Part 7: Visualization Guide

| File | Shows | Key Concepts |
|------|-------|--------------|
| `viz_1_sigmoid.png` | σ(z) and σ'(z) | Squashing, vanishing gradients |
| `viz_2_cost_function.png` | -log(h) and -log(1-h) | Why wrong predictions are penalized |
| `viz_3_gradient_descent.png` | θ updates | Learning rate effects |
| `viz_4_one_vs_all.png` | 4 classifiers | Binary decomposition |
| `viz_5_feature_importance.png` | θⱼ values | Which subjects matter |
| `viz_6_confusion_matrix.png` | Predictions vs actual | Model accuracy |
| `viz_7_probability_distribution.png` | P(house) histograms | Classifier confidence |
| `viz_8_normalization.png` | Before/after scaling | Why normalize |
| `viz_9_decision_boundaries.png` | Class regions | How model separates |

---

# Part 8: Common Defense Questions

## Q: What does each symbol in the cost function mean?
**A**: 
- J(θ) = total cost (function of weights θ)
- m = number of training samples (1600)
- Σ = sum over all samples
- y = true label (0 or 1)
- h = predicted probability
- log = natural logarithm

## Q: Why is there a negative sign in log loss?
**A**: log(x) for x<1 is negative. The negative sign makes cost positive.

## Q: What's the difference between θ and x?
**A**: 
- θ = learned weights (what we train)
- x = input features (given data)

## Q: What does (1/m) do in the gradient?
**A**: Averages the gradient over all samples. Makes learning rate independent of dataset size.

## Q: Why θᵀx and not θ·x?
**A**: θᵀ transposes θ to a row vector so we can dot product with column vector x.

## Q: What's an epoch vs an iteration?
**A**: 
- Iteration = one gradient update
- Epoch = one pass through entire dataset
- For batch GD: 1 epoch = 1 iteration
- For SGD: 1 epoch = m iterations

---

# Part 9: Accuracy Results

```
Training Accuracy: 98.19%
Correct: 1571/1600

Per-house breakdown:
  Gryffindor: 318/327 (97.2%)
  Hufflepuff: 525/529 (99.2%)
  Ravenclaw:  435/443 (98.2%)
  Slytherin:  293/301 (97.3%)
```

**Exceeds 98% requirement! ✅**