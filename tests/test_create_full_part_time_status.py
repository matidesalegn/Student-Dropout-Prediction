import pandas as pd
from src.data.clean_data import create_full_part_time_status

def test_create_full_part_time_status():
    data = {
        'Curricular units 1st sem (enrolled)': [3, 6, 4],
        'Curricular units 2nd sem (enrolled)': [2, 4, 5]
    }
    df = pd.DataFrame(data)

    df_with_status = create_full_part_time_status(df)
    
    assert 'Full-time/Part-time' in df_with_status.columns
    assert df_with_status['Full-time/Part-time'].tolist() == [0, 1, 1]
