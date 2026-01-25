from utils import get_numeric_cols, load_model, sigmoid, arr_tofloat
import argparse

def main() -> int:
    """"""
    parser = argparse.ArgumentParser(description="DSLR Logistic Regression Predictor (one-vs-rest)")
    parser.add_argument("test_csv", help="Path to dataset_test.csv")
    parser.add_argument("model_json", help="Path to model.json produced by logreg_train.py")
    parser.add_argument("--out", default="houses.csv", help="Output predictions file (houses.csv)")
    args = parser.parse_args()
    try:
        model = load_model("model.json")
    except Exception:
        return 1
    thetas_dict = model["thetas_dict"]
    classes = model["classes"]
    subjects = model["subjects"]
    mu = model["mu"]
    sigma = model["sigma"]

    data = load()


    print("What is the mileage of your car?")
    mileage = input()
    try:
        mileage = float(mileage)

        if mileage < 0:
            print("Mileage cannot be negative.")
            return 1

        mileage_original = mileage
        if model["normalized"]:
            if model["x_min"] is None or model["x_max"] is None:
                print("Config file is missing normalization params")
                return 1
            mileage = normalize_data(mileage, float(data["x_min"]),
                                     float(data["x_max"]))

        estimated_price = estimate_price(mileage, theta0, theta1)
        toprint = f"Estimated price for a car with {mileage_original:.0f}"
        toprint += f"km mileage is: ${estimated_price:.2f}"
        print(toprint)
        return 0
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
    return 1


if __name__ == "__main__":
    main()
