# src/app/streamlit_app.py

import streamlit as st
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Add src to the path to allow relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the data processing modules
from data.load_data import load_dataset
from data.clean_data import clean_column_names, handle_missing_values, detect_outliers, cap_outliers
from data.transform_data import encode_categorical_variables, scale_features, feature_engineering

def main():
    st.title("Student Dropout Prediction")

    # Set the path to the raw data CSV file
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
        num_cols = ['Admission grade', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
        outliers = detect_outliers(df, num_cols)
        if st.checkbox("Show outliers"):
            for col, outliers_col in outliers.items():
                st.write(f"Outliers in {col}:")
                st.write(outliers_col)

        # Visualize Outliers
        if st.checkbox("Show box plots for numerical columns"):
            st.write("Box plots for numerical columns:")
            fig, axes = plt.subplots(1, len(num_cols), figsize=(12, 8))
            for i, col in enumerate(num_cols):
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(col)
            st.pyplot(fig)

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

        # Perform Descriptive Statistics
        if st.checkbox("Show descriptive statistics"):
            st.write("Numerical Descriptive Statistics:")
            st.write(df.describe())

        # Check if there are any categorical (object) columns before attempting to describe them
        categorical_cols = df.select_dtypes(include=['object', 'category'])
        if not categorical_cols.empty:
            st.write("Categorical Descriptive Statistics:")
            st.write(df.describe(include='object'))
        else:
            st.write("No categorical columns available for descriptive statistics.")

        # Correlation Analysis
        if st.checkbox("Show correlation matrix heatmap"):
            numeric_df = df.select_dtypes(include=['number'])
            correlation_matrix = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

        # Hypothesis Testing: T-test for Admission Grade (Dropout vs Graduate)
        dropouts = df[df['Target'] == 1]['Admission grade']
        graduates = df[df['Target'] == 0]['Admission grade']
        t_stat, p_value = stats.ttest_ind(dropouts, graduates)

        if st.checkbox("Show t-test results for Admission Grade (Dropout vs Graduate)"):
            st.write(f"T-test results:\nT-statistic: {t_stat}, P-value: {p_value}")

        # Hypothesis Testing: Chi-square test for Gender vs Target
        if st.checkbox("Show chi-square test results for Gender vs Dropout"):
            contingency_table = pd.crosstab(df['Gender'], df['Target'])
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-square test results for Gender vs Dropout:\nChi-square stat: {chi2_stat}, P-value: {p_val}")
    else:
        st.error("Failed to load data.")

if __name__ == '__main__':
    main()