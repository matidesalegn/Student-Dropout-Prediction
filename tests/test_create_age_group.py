import pandas as pd
from src.data.clean_data import create_age_group

def test_create_age_group():
    data = {'Age at enrollment': [17, 20, 30, 40, 50]}
    df = pd.DataFrame(data)

    df_with_age_group = create_age_group(df)
    
    assert 'Age Group' in df_with_age_group.columns
    assert df_with_age_group['Age Group'].tolist() == [0, 1, 2, 3, 4]
