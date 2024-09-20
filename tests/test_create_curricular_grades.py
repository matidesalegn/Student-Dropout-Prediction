import pandas as pd
from src.data.clean_data import create_curricular_grades

def test_create_curricular_grades():
    data = {
        'Curricular units 1st sem (grade)': [10, 12, 14],
        'Curricular units 2nd sem (grade)': [11, 13, 15]
    }
    df = pd.DataFrame(data)

    df_with_grades = create_curricular_grades(df)
    
    assert 'Total curricular grade' in df_with_grades.columns
    assert df_with_grades['Total curricular grade'].tolist() == [21, 25, 29]
