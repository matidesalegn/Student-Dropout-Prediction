# tests/test_clean_data.py
import unittest
import pandas as pd
from src.data.clean_data import clean_column_names, handle_missing_values

class TestCleanData(unittest.TestCase):

    def test_clean_column_names(self):
        data = {' \t Name': [1, 2, 3], ' \t Age': [10, 20, 30]}
        df = pd.DataFrame(data)
        df_cleaned = clean_column_names(df)
        self.assertIn('Name', df_cleaned.columns)
        self.assertIn('Age', df_cleaned.columns)

    def test_handle_missing_values_drop(self):
        data = {'A': [1, 2, None], 'B': [None, 2, 3]}
        df = pd.DataFrame(data)
        df_cleaned = handle_missing_values(df, method='drop')
        self.assertEqual(df_cleaned.isnull().sum().sum(), 0)
        self.assertEqual(len(df_cleaned), 1)

if __name__ == '__main__':
    unittest.main()
