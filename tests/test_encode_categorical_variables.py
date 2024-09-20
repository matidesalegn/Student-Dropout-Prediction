import pandas as pd
from src.data.clean_data import encode_categorical_variables

def test_encode_categorical_variables_label():
    data = {'Gender': ['Male', 'Female', 'Female', 'Male'], 'Target': ['Yes', 'No', 'Yes', 'No']}
    df = pd.DataFrame(data)
    
    # Test label encoding
    df_encoded = encode_categorical_variables(df, columns=['Gender', 'Target'], encoding_type='label')
    
    assert df_encoded['Gender'].dtype == 'int64'
    assert df_encoded['Target'].dtype == 'int64'
    assert df_encoded['Gender'].unique().tolist() == [1, 0]  # Assuming Male=1, Female=0
    assert df_encoded['Target'].unique().tolist() == [1, 0]  # Assuming Yes=1, No=0

def test_encode_categorical_variables_onehot():
    data = {'Gender': ['Male', 'Female', 'Female', 'Male'], 'Target': ['Yes', 'No', 'Yes', 'No']}
    df = pd.DataFrame(data)
    
    # Test one-hot encoding
    df_encoded = encode_categorical_variables(df, columns=['Gender'], encoding_type='onehot')
    
    assert 'Gender_Male' in df_encoded.columns
    assert 'Gender_Female' in df_encoded.columns
    assert df_encoded['Gender_Male'].sum() == 2
    assert df_encoded['Gender_Female'].sum() == 2
