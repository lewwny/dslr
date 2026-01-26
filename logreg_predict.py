from load import load
import numpy as np
from utils import load_model, sigmoid, preprocess_vals, add_bias
from typing import List
import argparse

def houses_tocsv(path: str, preds: List[str]) -> None:
    """writes into csv list of predictions"""
    with open(path, "w", encoding="utf-8") as file:
        file.write("Index,Hogwarts House\n")
        for index, house in enumerate(preds):
            file.write(f"{index},{house}\n")

def main() -> int:
    """loads model, processes vals and writes predictions to csv"""
    parser = argparse.ArgumentParser(description="DSLR Logistic Regression Predictor (one-vs-rest)")
    parser.add_argument("data_csv", help="Path to dataset_test.csv")
    parser.add_argument("--out", default="houses.csv", help="Output predictions file (houses.csv)")
    args = parser.parse_args()

    try:
        model = load_model("model.json")
        thetas_dict = model["thetas_dict"]
        houses = model["houses"]
        subjects = model["subjects"]
        mu = model["mu"]
        sigma = model["sigma"]

        data = load(args.data_csv)
        if data is None:
            raise ValueError("Could not load the data csv")

        vals = preprocess_vals(data, subjects, mu, sigma)
        biased_vals = add_bias(vals)

        probas = []
        for house in houses:
            theta = np.array(thetas_dict[house], dtype=float)
            proba = sigmoid(biased_vals @ theta)
            probas.append(proba)

        pstack  = np.vstack(probas).T
        indices = np.argmax(pstack, axis=1)
        preds = [houses[int(i)] for i in indices]

        houses_tocsv(args.out, preds)
        print(f"Wrote predictions to: {args.out}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
