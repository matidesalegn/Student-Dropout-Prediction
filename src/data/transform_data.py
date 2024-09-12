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
    print("Available columns in the DataFrame:", df.columns)

    # If no columns are specified, use default columns 'Target' and 'Gender'
    if columns is None:
        columns = ['Target', 'Gender']

    # Loop through specified columns and check if they are already encoded
    for col in columns:
        if col in df.columns:
            # Skip encoding if the column is already numeric (assuming it has been encoded)
            if df[col].dtype in ['int64', 'float64']:
                print(f"Skipping encoding for '{col}' as it is already numeric.")
                continue

            # Perform encoding if column is categorical
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
    if 'Age at enrollment' in df.columns:
        df['Age Group'] = pd.cut(df['Age at enrollment'],
                                 bins=[0, 18, 25, 35, 45, 100],
                                 labels=['<18', '18-25', '26-35', '36-45', '45+'])
        # Optional: Encoding Age Group if needed
        age_group_mapping = {'<18': 0, '18-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
        df['Age Group'] = df['Age Group'].map(age_group_mapping)
    else:
        raise KeyError("'Age at enrollment' column is missing.")
    
    return df

def create_curricular_grades(df):
    """
    Create 'Total curricular grade' derived feature.

    Parameters:
    - df (DataFrame): DataFrame containing 1st and 2nd semester grades.

    Returns:
    - df (DataFrame): DataFrame with the 'Total curricular grade' feature.
    """
    if 'Curricular units 1st sem (grade)' in df.columns and 'Curricular units 2nd sem (grade)' in df.columns:
        df['Total curricular grade'] = df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']
    else:
        raise KeyError("Missing required columns for calculating 'Total curricular grade'.")
    
    return df

def create_full_part_time_status(df):
    """
    Create 'Full-time/Part-time' derived feature based on the number of curricular units enrolled.

    Parameters:
    - df (DataFrame): DataFrame containing curricular units enrollment.

    Returns:
    - df (DataFrame): DataFrame with the 'Full-time/Part-time' feature.
    """
    if 'Curricular units 1st sem (enrolled)' in df.columns and 'Curricular units 2nd sem (enrolled)' in df.columns:
        df['Full-time/Part-time'] = df.apply(
            lambda x: 1 if (x['Curricular units 1st sem (enrolled)'] + x['Curricular units 2nd sem (enrolled)']) >= 6 else 0, axis=1)
    else:
        raise KeyError("Missing required columns for calculating 'Full-time/Part-time' status.")
    
    return df

def feature_engineering(df):
    """
    Create derived features such as 'Age Group', 'Total curricular grade', and 'Full-time/Part-time'.
    
    Parameters:
    - df (DataFrame): DataFrame containing the data.
    
    Returns:
    - df (DataFrame): DataFrame with new features.
    """
    # Print columns for debugging
    print("Columns in DataFrame for feature engineering:", df.columns)

    # Define required columns and check for missing ones
    required_columns = ['Age at enrollment', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns for feature engineering: {missing_cols}")
    
    # Create individual derived features only if the required columns are present
    if 'Age at enrollment' in df.columns:
        df = create_age_group(df)
    if 'Curricular units 1st sem (grade)' in df.columns and 'Curricular units 2nd sem (grade)' in df.columns:
        df = create_curricular_grades(df)
    if 'Curricular units 1st sem (enrolled)' in df.columns and 'Curricular units 2nd sem (enrolled)' in df.columns:
        df = create_full_part_time_status(df)
    
    return df

def scale_features(df, scale_cols):
    """
    Scale the features in the DataFrame.
    
    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - scale_cols (list): List of columns to scale.
    
    Returns:
    - df (DataFrame): DataFrame with scaled features.
    """
    # Print columns for debugging
    print("Columns in DataFrame for scaling:", df.columns)

    # Check for missing columns
    missing_cols = [col for col in scale_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")

    # Perform scaling (example with MinMaxScaler)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df