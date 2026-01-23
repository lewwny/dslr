import os
import json


def load_thetas(path: str) -> dict[str, any]:
    """loads thetas from a json file"""
    if not os.path.exists(path):
        return {
            "thetaGryf": 0.0,
            "thetaHuff": 0.0,
            "thetaSlit": 0.0,
            "thetaRave": 0.0,
            "normalized": False,
            "x_min": None,
            "x_max": None
        }
    try:
        with open(path, 'r') as file:
            data = json.load(file)
            return {
                "theta0": float(data.get("theta0", 0.0)),
                "theta1": float(data.get("theta1", 0.0)),
                "normalized": bool(data.get("normalized", False)),
                "x_min": data.get("x_min", None),
                "x_max": data.get("x_max", None),
            }
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: could not read {path}: {e}")
        return None


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """estimates price of a car with mileage using linear regression"""
    return theta0 + (theta1 * mileage)


def normalize_data(mileage: float, x_min: float, x_max: float) -> float:
    """normalizer for the data using xmin and xmax"""
    if x_min == x_max:
        return 0.0
    return (mileage - x_min) / (x_max - x_min)


def main() -> int:
    """prompts for a mileage and calculates estimate"""
    try:
        data = load_thetas("thetas.json")
    except Exception:
        return 1
    theta0 = data["theta0"]
    theta1 = data["theta1"]

    print("What is the mileage of your car?")
    mileage = input()
    try:
        mileage = float(mileage)

        if mileage < 0:
            print("Mileage cannot be negative.")
            return 1

        mileage_original = mileage
        if data["normalized"]:
            if data["x_min"] is None or data["x_max"] is None:
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
