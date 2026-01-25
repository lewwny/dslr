import numpy as np
import argparse
from typing import List, Dict
from utils import sigmoid, preprocess_vals, load_model, add_bias
from load import load


def predict_classes(col: np.ndarray, classes: List[str], thetas: Dict[str, List[float]]) -> List[str]:
    probas = []
    for house in classes:
        theta = np.array(thetas[house], dtype=float)
        proba = sigmoid(col @ theta)
        probas.append(proba)

    pstack = np.vstack(probas).T
    indices = np.argmax(pstack, axis=1)
    return [classes[int(i)] for i in indices]


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = 0
    size = len(y_true)
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return (correct / size) if size > 0 else 0.0


def confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str]) -> np.ndarray:
    size = len(classes)
    indices = {clas: i for i, clas in enumerate(classes)}
    cmatrix = np.zeros((size, size), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in indices and pred in indices:
            cmatrix[indices[true], indices[pred]] +=1
    return cmatrix


def main() -> int:
    """"""
    parser = argparse.ArgumentParser(description="Evaluate the DSLR model (accuracy + confusion matrix)")
    parser.add_argument("data_csv", help="Path to dataset_test.csv")
    parser.add_argument("--out", default="houses.csv", help="Output predictions file (houses.csv)")
    args = parser.parse_args()

    try:
        model = load_model("model.json")
        thetas = model["thetas_dict"]
        classes = model["classes"]
        subjects = model["subjects"]
        mu = model["mu"]
        sigma = model["sigma"]

        data = load(args.data_csv)
        if data is None:
            raise ValueError("Could not load the data csv")
        if "Hogwarts Houses" not in data.columns:
            raise ValueError("Data csv must contain Hogwarts Houses")

        vals = preprocess_vals(data, subjects, mu, sigma)
        biased_vals = add_bias(vals)

        y_true = [str(val) for val in data["Hogwarts Houses"].to_numpy()]
        y_pred = predict_classes(biased_vals, classes, thetas)

        acc = compute_accuracy(y_true, y_pred)
        print(f"Accuracy: {acc*100:.2f}%\n")

        cm = confusion_matrix(y_true, y_pred, classes)
        print("Confusion Matrix (rows=true, cols=pred):")
        header = " " * 14 + " ".join([f"{clas[:10]:>10}" for clas in classes])
        print(header)
        for index, clas in enumerate(classes):
            row = " ".join([f"{cm[index, clas]:>10d}" for i in range(len(classes))])
            print(f"{clas[:12]:>12}  {row}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
