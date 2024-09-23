import unittest
import os
import pandas as pd
from src.data.load_data import load_dataset

class TestLoadData(unittest.TestCase):

    def test_load_dataset_file_exists(self):
        # Replace with your actual CSV path or use a test file
        test_file = '../data/raw/data.csv'
        df = load_dataset(test_file)
        self.assertIsNotNone(df)

    def test_load_dataset_file_does_not_exist(self):
        # Non-existent file
        df = load_dataset('data/raw/non_existent.csv')
        self.assertIsNone(df)

if __name__ == '__main__':
    unittest.main()