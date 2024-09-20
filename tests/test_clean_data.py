import pandas as pd
from src.data.clean_data import clean_column_names, handle_missing_values

def test_clean_column_names():
    data = {' \t Name': [1, 2, 3], ' \t Age': [10, 20, 30]}
    df = pd.DataFrame(data)
    df_cleaned = clean_column_names(df)
    assert 'Name' in df_cleaned.columns
    assert 'Age' in df_cleaned.columns

def test_handle_missing_values_drop():
    data = {'A': [1, 2, None], 'B': [None, 2, 3]}
    df = pd.DataFrame(data)
    df_cleaned = handle_missing_values(df, method='drop')
    assert df_cleaned.isnull().sum().sum() == 0
    assert len(df_cleaned) == 1