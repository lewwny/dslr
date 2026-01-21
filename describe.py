from load import load
import pandas as pd
import sys

def counter(data):
    """Count function"""
    count = len(data)
    return count

def meaner(data):
    """Mean function"""
    count = counter(data)
    total = 0
    for value in data:
        total += value
    mean = total / count if count != 0 else 0
    return mean

def stder(data):
    """Standard Deviation function"""
    mean = meaner(data)
    squared_diffs = (data - mean) ** 2
    total_sq_diff = 0
    for value in squared_diffs:
        total_sq_diff += value
    variance = total_sq_diff / (counter(data) - 1) if counter(data) > 1 else 0
    std_dev = variance ** 0.5
    return std_dev

def min_func(data):
    """Min function"""
    if len(data) == 0:
        return 0
    min_value = data[0]
    for value in data:
        if value < min_value:
            min_value = value
    return min_value

def quantiler(data, q):
    """Quantile function"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0
    pos = (n - 1) * q
    lower_index = int(pos)
    upper_index = lower_index + 1
    if upper_index >= n:
        return sorted_data[lower_index]
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index]
    interpolated = lower_value + (upper_value - lower_value) * (pos - lower_index)
    return interpolated

def max_func(data):
    """Max function"""
    if len(data) == 0:
        return 0
    max_value = data[0]
    for value in data:
        if value > max_value:
            max_value = value
    return max_value

def ft_describe(data):
    """Describe function"""
    new_data = pd.DataFrame()
    for col in data.columns:
        if col in ['Index', 'Hogwarts House']:
            continue
        if data[col].dtype not in ['int64', 'float64']:
            continue
        col_data = data[col].dropna()
        count = counter(col_data)
        mean = meaner(col_data)
        std = stder(col_data)
        min_val = min_func(col_data)
        q1 = quantiler(col_data, 0.25)
        median = quantiler(col_data, 0.50)
        q3 = quantiler(col_data, 0.75)
        max_val = max_func(col_data)
        new_data[col] = [count, mean, std, min_val, q1, median, q3, max_val]
    new_data.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return new_data

def main():
    """main func"""
    try:
        if len(sys.argv) != 2:
            raise ValueError("Please provide exactly one argument: the path to the CSV file.")
        path = sys.argv[1]
        data = load(path)
        print(ft_describe(data))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()