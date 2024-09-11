# src/data/clean_data.py

import pandas as pd
import numpy as np

def clean_column_names(df):
    """
    Remove any tabs or whitespaces from column names.

    Parameters:
    - df (DataFrame): DataFrame with original column names.

    Returns:
    - df (DataFrame): DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.replace('\t', '').str.strip()
    return df

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.

    Parameters:
    - df (DataFrame): Original DataFrame.

    Returns:
    - df (DataFrame): DataFrame with missing values handled.
    """
    # Since no missing values are detected, this function can be expanded if needed.
    return df

def detect_outliers(df, num_cols):
    """
    Detect outliers in numerical columns using the IQR method.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - num_cols (list): List of numerical columns to check for outliers.

    Returns:
    - outliers_dict (dict): Dictionary with columns as keys and outlier indices as values.
    """
    outliers_dict = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outliers_dict[col] = outliers
    return outliers_dict

def cap_outliers(df, cols_with_outliers):
    """
    Cap outliers to the 1st and 99th percentiles.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - cols_with_outliers (list): List of columns to cap outliers.

    Returns:
    - df (DataFrame): DataFrame with outliers capped.
    """
    for col in cols_with_outliers:
        lower_cap = df[col].quantile(0.01)
        upper_cap = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
    return df
