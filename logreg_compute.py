import numpy as np
import argparse
from typing import List, Dict
from utils import sigmoid, preprocess_vals, load_model, add_bias
from load import load


def predict_houses(col: np.ndarray, houses: List[str], thetas: Dict[str, List[float]]) -> List[str]:
    """uses the model values to make prediction y values"""
    probas = []
    for house in houses:
        theta = np.array(thetas[house], dtype=float)
        proba = sigmoid(col @ theta)
        probas.append(proba)

    pstack = np.vstack(probas).T
    indices = np.argmax(pstack, axis=1)
    return [houses[int(i)] for i in indices]


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """evaluates model accuracy for making predictions"""
    correct = 0
    size = len(y_true)
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return (correct / size) if size > 0 else 0.0


def confusion_matrix(y_true: List[str], y_pred: List[str], houses: List[str]) -> np.ndarray:
    """calculates the confusion matrix value for the model used with preds/true"""
    size = len(houses)
    indices = {house: i for i, house in enumerate(houses)}
    cmatrix = np.zeros((size, size), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in indices and pred in indices:
            cmatrix[indices[true], indices[pred]] +=1
    return cmatrix


def main() -> int:
    """loads model and calculates accuracy and confusion matrix"""
    parser = argparse.ArgumentParser(description="Evaluate the DSLR model (accuracy + confusion matrix)")
    parser.add_argument("data_csv", help="Path to dataset_test.csv")
    parser.add_argument("--out", default="houses.csv", help="Output predictions file (houses.csv)")
    args = parser.parse_args()

    try:
        model = load_model("model.json")
        thetas = model["thetas_dict"]
        houses = model["houses"]
        subjects = model["subjects"]
        mu = model["mu"]
        sigma = model["sigma"]

        data = load(args.data_csv)
        if data is None:
            raise ValueError("Could not load the data csv")
        if "Hogwarts House" not in data.columns:
            raise ValueError("Data csv must contain Hogwarts House")

        vals = preprocess_vals(data, subjects, mu, sigma)
        biased_vals = add_bias(vals)

        y_true = [str(val) for val in data["Hogwarts House"].to_numpy()]
        y_pred = predict_houses(biased_vals, houses, thetas)

        acc = compute_accuracy(y_true, y_pred)
        print(f"Accuracy: {acc*100:.2f}%\n")

        cm = confusion_matrix(y_true, y_pred, houses)
        print("Confusion Matrix (rows=true, cols=pred):")
        header = " " * 14 + " ".join([f"{house[:10]:>10}" for house in houses])
        print(header)
        for index, house in enumerate(houses):
            row = " ".join([f"{cm[index, house]:>10d}" for house in range(len(houses))])
            print(f"{house[:12]:>12}  {row}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
