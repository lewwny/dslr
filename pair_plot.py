import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from load import load
import sys

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by removing non-numeric columns except 'Hogwarts House'"""
    numeric_df = df.select_dtypes(include=['number'])
    if 'Hogwarts House' in df.columns:
        numeric_df['Hogwarts House'] = df['Hogwarts House']
    return numeric_df.drop(columns=['Index'], errors='ignore')


def main():
    """main func"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide exactly one argument: the path to the CSV file.")
        path = sys.argv[1]
        data = load(path)
        df = data.dropna()
        data = clean_data(df)
        sns.pairplot(data, hue="Hogwarts House", diag_kind="hist")
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()