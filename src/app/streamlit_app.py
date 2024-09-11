# src/app/streamlit_app.py

import streamlit as st
import os
import sys

# Ensure the 'src' directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now we can use absolute imports
from data.load_data import load_dataset
from data.clean_data import clean_column_names, handle_missing_values, detect_outliers, cap_outliers
from data.transform_data import encode_categorical_variables, scale_features, feature_engineering

def main():
    st.title("Student Dropout Prediction")

    # Use absolute path for the dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/data.csv'))

    # Load data
    df = load_dataset(data_path)

    if df is not None:
        st.success("Data loaded successfully!")

        # Data Cleaning
        df = clean_column_names(df)
        df = handle_missing_values(df)

        # Display raw data
        if st.checkbox("Show raw data"):
            st.write(df.head())

        # Outlier Detection
        num_cols = ['Admission grade', 'Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (grade)']
        outliers = detect_outliers(df, num_cols)
        if st.checkbox("Show outliers"):
            st.write(outliers)

        # Outlier Capping
        df = cap_outliers(df, num_cols)

        # Feature Engineering
        df = encode_categorical_variables(df)
        df = feature_engineering(df)

        # Scaling Numerical Features
        scale_cols = ['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP',
                      'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
        df = scale_features(df, scale_cols)

        # Display preprocessed data
        if st.checkbox("Show preprocessed data"):
            st.write(df.head())

        # Placeholder for model training and prediction
        st.write("Model training and prediction functionality will be added here.")
    else:
        st.error("Failed to load data.")

if __name__ == '__main__':
    main()
