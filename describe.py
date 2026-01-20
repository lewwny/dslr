from load import load_csv
import sys

def main():
    """main func"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide exactly one argument: the path to the CSV file.")
        path = sys.argv[1]
        data = load_csv(path)
        print(data.describe())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()