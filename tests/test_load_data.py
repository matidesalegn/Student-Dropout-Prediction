import os
import pandas as pd
from src.data.load_data import load_dataset

def test_load_dataset_file_exists():
    # Replace with your actual CSV path or use a test file
    test_file = '../data/raw/data.csv'
    df = load_dataset(test_file)
    assert df is not None

def test_load_dataset_file_does_not_exist():
    # Non-existent file
    df = load_dataset('data/raw/non_existent.csv')
    assert df is None