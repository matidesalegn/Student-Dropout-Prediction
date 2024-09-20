import pandas as pd
from src.data.clean_data import scale_features

def test_scale_features():
    data = {'Grade 1': [10, 20, 30], 'Grade 2': [40, 50, 60]}
    df = pd.DataFrame(data)
    
    df_scaled = scale_features(df, scale_cols=['Grade 1', 'Grade 2'])
    
    # Check if values are between 0 and 1 after scaling
    assert df_scaled['Grade 1'].min() >= 0 and df_scaled['Grade 1'].max() <= 1
    assert df_scaled['Grade 2'].min() >= 0 and df_scaled['Grade 2'].max() <= 1
