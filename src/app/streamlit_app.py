#src/app/streamlit_app.py
import streamlit as st
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # For loading saved models

# Add src to the path to allow relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the data processing modules
from data.load_data import load_dataset
from data.clean_data import clean_column_names, handle_missing_values, detect_outliers, cap_outliers
from data.transform_data import encode_categorical_variables, scale_features, feature_engineering

# Load the models
def load_models():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))

    models = {
        'Logistic Regression': joblib.load(os.path.join(model_dir, 'Logistic_Regression.joblib')),
        'Random Forest': joblib.load(os.path.join(model_dir, 'Random_Forest.joblib')),
        'Gradient Boosting': joblib.load(os.path.join(model_dir, 'Gradient_Boosting.joblib'))
    }
    return models

def main():
    st.title("Student Dropout Prediction")

    # Load models
    models = load_models()

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

        # Initial Data Exploration
        if st.checkbox("Show initial data exploration"):
            st.write("### Dataset Shape:")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            st.write("### Data Types:")
            st.write(df.dtypes)

            st.write("### Summary Statistics:")
            st.write(df.describe())

        # Check for missing values
        if st.checkbox("Check for missing values"):
            st.write("### Missing Values per Column:")
            missing_values = df.isnull().sum()
            st.write(missing_values[missing_values > 0])

            if missing_values.sum() > 0:
                st.write("### Missing Values Heatmap:")
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
                st.pyplot(plt.gcf())

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


        # Feature Importance Using RandomForest
        # Prepare data for modeling
        X = df.drop(columns='Target')
        y = df['Target']

        # Split the data for training (80%) and testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Importance Using RandomForest
        # Only show feature importance if the checkbox is selected
        if st.checkbox("Show feature importance"):
            st.write("### Feature Importance using RandomForest:")
            
            # Train a RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Calculate feature importance
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            # Display feature importance
            st.write(feature_importance_df)

            # Plot the feature importance
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
            st.pyplot(fig)
        # Perform Descriptive Statistics
        if st.checkbox("Show descriptive statistics"):
            st.write("Numerical Descriptive Statistics:")
            st.write(df.describe())

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

        # Model Prediction Interface
        st.write("### Model Prediction")

        # Extract the feature names from the training data
        feature_names = X_train.columns.tolist()

        # Create a form for user input
        with st.form(key='prediction_form'):
            st.write("Enter details for prediction:")

            # Add form fields (update these fields based on your feature set)
            admission_grade = st.number_input("Admission Grade")
            curricular_units_1st_sem = st.number_input("Curricular Units 1st Sem Grade")
            curricular_units_2nd_sem = st.number_input("Curricular Units 2nd Sem Grade")
            age_group = st.selectbox("Age Group", options=['<18', '18-25', '26-35', '36-45', '45+'])
            # Include other fields as needed

            # Convert age group to numeric
            age_group_mapping = {'<18': 0, '18-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
            age_group_numeric = age_group_mapping.get(age_group, -1)

            # Submit button
            submit_button = st.form_submit_button("Predict")

            if submit_button:
                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    'Admission grade': [admission_grade],
                    'Curricular units 1st sem (grade)': [curricular_units_1st_sem],
                    'Curricular units 2nd sem (grade)': [curricular_units_2nd_sem],
                    'Age Group': [age_group_numeric]
                    # Include other features as needed
                })

                # Apply the same preprocessing as used for training
                input_data = encode_categorical_variables(input_data)
                input_data = feature_engineering(input_data)
                input_data = scale_features(input_data, scale_cols)

                # Ensure the input data matches the training features
                input_data = input_data.reindex(columns=feature_names, fill_value=0)

                # Make predictions with all models
                predictions = {}
                for name, model in models.items():
                    pred = model.predict(input_data)
                    predictions[name] = pred[0]

                # Display predictions
                st.write("### Prediction Results")
                for model_name, prediction in predictions.items():
                    st.write(f"{model_name} Prediction: {prediction}")

    else:
        st.error("Failed to load data.")

if __name__ == '__main__':
    main()
