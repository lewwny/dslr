import pandas as pd


def load(path: str) -> pd.DataFrame:
    """loads a csv file from (path) and
    returns it as a Dataset"""
    try:
        dataset = pd.read_csv(path)
        rows, cols = dataset.shape
        print(f"Loading dataset of dimensions ({rows}, {cols})")
        return dataset
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError, ValueError):
        return None
    except Exception:
        return None