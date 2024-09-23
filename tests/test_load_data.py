# tests/test_load_data.py
import unittest
import os
from src.data.load_data import load_dataset

class TestLoadData(unittest.TestCase):

    def test_load_dataset_file_exists(self):
        # Dynamically construct the path relative to the current file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(base_dir, '../data/raw/data.csv')
        
        # Check if the file exists before running the test
        if not os.path.exists(test_file):
            self.fail(f"Test file {test_file} does not exist.")
        
        df = load_dataset(test_file)
        self.assertIsNotNone(df)

    def test_load_dataset_file_does_not_exist(self):
        # Testing for a non-existent file
        df = load_dataset('data/raw/non_existent.csv')
        self.assertIsNone(df)

if __name__ == '__main__':
    unittest.main()