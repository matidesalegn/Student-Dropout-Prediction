# src/data/load_data.py

import pandas as pd
import os

def load_dataset(file_path, delimiter=';', encoding='utf-8'):
    """
    Load dataset from a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.
    - delimiter (str): Delimiter used in the CSV file (default is ';').
    - encoding (str): Encoding of the CSV file (default is 'utf-8').

    Returns:
    - df (DataFrame): Loaded DataFrame or None if loading fails.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None

    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
