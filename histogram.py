import sys
from load import load_csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def draw_histogram():
    """function to draw the histogram for the courses"""
    


def compute():
    """function that splits values by house in order to show
    'Which Hogwarts course has a homogeneous score distribution
    between all four houses?'"""


def main():
    """main func"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide exactly one argument: the path to the CSV file.")
        path = sys.argv[1]
        data = load_csv(path)
        compute(data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()