# src/data/transform_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_categorical_variables(df):
    """
    Encode categorical variables, such as 'Target' and 'Gender'.

    Parameters:
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - df (DataFrame): DataFrame with encoded categorical variables.
    """
    # Encode the 'Target' column
    df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 2})
    df['Target'] = df['Target'].fillna(0).astype(int)
    # Encode other categorical variables if necessary
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
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def feature_engineering(df):
    """
    Create derived features such as 'Age Group', 'Total curricular grade', and 'Full-time/Part-time'.

    Parameters:
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - df (DataFrame): DataFrame with new features.
    """
    # Create Age Group
    df['Age Group'] = pd.cut(df['Age at enrollment'],
                             bins=[0, 18, 25, 35, 45, 100],
                             labels=['<18', '18-25', '26-35', '36-45', '45+'])
    
    # Calculate total curricular units grades
    df['Total curricular grade'] = (df['Curricular units 1st sem (grade)'] +
                                    df['Curricular units 2nd sem (grade)'])
    
    # Create Full-time/Part-time feature
    df['Full-time/Part-time'] = df.apply(
        lambda x: 1 if (x['Curricular units 1st sem (enrolled)'] +
                        x['Curricular units 2nd sem (enrolled)']) >= 6 else 0, axis=1)
    
    # Encode 'Age Group' if needed
    age_group_mapping = {'<18': 0, '18-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
    df['Age Group'] = df['Age Group'].map(age_group_mapping)
    
    return df
