# src/data/transform_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def encode_categorical_variables(df, columns=None, encoding_type='label'):
    """
    Encode categorical variables in the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - columns (list): List of columns to encode. If None, encodes default columns like 'Target' and 'Gender'.
    - encoding_type (str): Encoding method to use, either 'label' for Label Encoding or 'onehot' for One-Hot Encoding.

    Returns:
    - df (DataFrame): DataFrame with encoded categorical variables.
    """
    if columns is None:
        # Default encoding for 'Target' and 'Gender'
        df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 2})
        df['Target'] = df['Target'].fillna(0).astype(int)
    else:
        # Encode specified columns
        for col in columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if encoding_type == 'label':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                elif encoding_type == 'onehot':
                    df = pd.get_dummies(df, columns=[col], prefix=col)
    return df

def scale_features(df, num_cols):
    """
    Scale numerical features using StandardScaler.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - num_cols (list): List of numerical columns to scale.

    Returns:
    - df (DataFrame): DataFrame with scaled features.
    """
    if not all(col in df.columns for col in num_cols):
        missing_cols = [col for col in num_cols if col not in df.columns]
        raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def create_age_group(df):
    """
    Create 'Age Group' derived feature.

    Parameters:
    - df (DataFrame): DataFrame containing 'Age at enrollment'.

    Returns:
    - df (DataFrame): DataFrame with the 'Age Group' feature.
    """
    df['Age Group'] = pd.cut(df['Age at enrollment'],
                             bins=[0, 18, 25, 35, 45, 100],
                             labels=['<18', '18-25', '26-35', '36-45', '45+'])
    # Optional: Encoding Age Group if needed
    age_group_mapping = {'<18': 0, '18-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
    df['Age Group'] = df['Age Group'].map(age_group_mapping)
    return df

def create_curricular_grades(df):
    """
    Create 'Total curricular grade' derived feature.

    Parameters:
    - df (DataFrame): DataFrame containing 1st and 2nd semester grades.

    Returns:
    - df (DataFrame): DataFrame with the 'Total curricular grade' feature.
    """
    df['Total curricular grade'] = df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']
    return df

def create_full_part_time_status(df):
    """
    Create 'Full-time/Part-time' derived feature based on the number of curricular units enrolled.

    Parameters:
    - df (DataFrame): DataFrame containing curricular units enrollment.

    Returns:
    - df (DataFrame): DataFrame with the 'Full-time/Part-time' feature.
    """
    df['Full-time/Part-time'] = df.apply(
        lambda x: 1 if (x['Curricular units 1st sem (enrolled)'] + x['Curricular units 2nd sem (enrolled)']) >= 6 else 0, axis=1)
    return df

def feature_engineering(df):
    """
    Create derived features such as 'Age Group', 'Total curricular grade', and 'Full-time/Part-time'.

    Parameters:
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - df (DataFrame): DataFrame with new features.
    """
    # Create individual derived features
    df = create_age_group(df)
    df = create_curricular_grades(df)
    df = create_full_part_time_status(df)
    
    return df