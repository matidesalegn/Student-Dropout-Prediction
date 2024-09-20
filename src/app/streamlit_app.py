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
# Define a function for univariate analysis
def univariate_analysis(df):
    st.subheader("Univariate Analysis")

    # Display histograms and box plots for numerical columns
    st.write("### Histograms")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        st.write(f"**{col}**")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], ax=ax[0], kde=True)
        ax[0].set_title(f"Histogram of {col}")
        sns.boxplot(y=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    # Bar charts for categorical columns
    st.write("### Bar Charts for Categorical Variables")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"**{col}**")
        st.bar_chart(df[col].value_counts())

# Define a function for bivariate analysis
def bivariate_analysis(df):
    st.subheader("Bivariate Analysis")

    # Correlation matrix
    st.write("### Correlation Matrix")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Scatter plots for pairs of numerical columns
    st.write("### Scatter Plot Pairs")
    pair_plot = sns.pairplot(df[numerical_columns])
    st.pyplot(pair_plot)

# Define a function for multivariate analysis (PCA)
def multivariate_analysis(df):
    st.subheader("Multivariate Analysis")

    # PCA plot
    st.write("### Principal Component Analysis (PCA)")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_scaled = StandardScaler().fit_transform(df[numerical_columns])
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)

    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Target'] = df['Target']  # Add target for coloring

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title='PCA of Dataset')
    st.plotly_chart(fig)

# Load the XGBoost model
def load_xgboost_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))
    xgboost_model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    return joblib.load(xgboost_model_path)
 
# Define a manual mapping for numeric predictions to human-readable labels
prediction_mapping = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

def main():
    st.title("Student Dropout Prediction")

    # Load XGBoost model
    xgboost_model = load_xgboost_model()

    # Set the path to the raw data CSV file
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/data.csv'))

    # Load data
    df = load_dataset(data_path)

    if df is not None:
        st.success("Data loaded successfully!")

        # Data Cleaning
        df = clean_column_names(df)
        df = handle_missing_values(df)

        # Remove the 'Target' column if it exists in the data (since it's the label, not a feature)
        feature_cols = df.drop(columns=['Target']).columns.tolist() if 'Target' in df.columns else df.columns.tolist()

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
 
        # Define columns for categorical and numerical inputs
        categorical_cols = [
            'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
            'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
            'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
        ]
 
        numerical_cols = [
            'Application order', 'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
            'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]
 
        # Create a form for user input
        with st.form(key='prediction_form'):
            st.write("Enter details for prediction:")
 
            # Input fields for categorical columns
            user_input = {}
            for col in categorical_cols:
                user_input[col] = st.selectbox(f"{col}", df[col].unique())
 
            # Input fields for numerical columns
            for col in numerical_cols:
                user_input[col] = st.number_input(f"{col}")
 
            # Submit button
            submit_button = st.form_submit_button("Predict")
 
            if submit_button:
                # Convert the input data to a DataFrame
                input_data = pd.DataFrame([user_input])
 
                # Preprocess the input data: Encoding, scaling, and feature engineering
                input_data = encode_categorical_variables(input_data)
                input_data = feature_engineering(input_data)
                input_data = scale_features(input_data, numerical_cols)
 
                # Ensure the input data matches the training features (excluding the 'Target' column)
                input_data = input_data.reindex(columns=feature_cols, fill_value=0)
 
                # Make predictions using the XGBoost model
                numeric_prediction = xgboost_model.predict(input_data)[0]
 
                # Use the manual mapping to convert the numeric prediction to human-readable label
                label_prediction = prediction_mapping.get(numeric_prediction, "Unknown")
 
                # Display prediction
                st.write(f"Prediction: {label_prediction}")

        # Display Insights, Recommendations, and Conclusions
        st.write("### Insights, Recommendations, and Conclusions")

        st.subheader("Insights")
        st.write("""
        - The dataset contains 4,424 rows and 37 columns.
        - Outliers were detected in key numerical variables such as Admission grade and semester grades.
        - The T-test showed no significant difference between dropouts and graduates in Admission grades.
        - Gender shows a significant impact on dropout rates according to the Chi-square test.
        """)

        st.subheader("Recommendations")
        st.write("""
        - Investigate missing values and handle them appropriately to improve model performance.
        - Consider outlier treatment techniques like transformations to avoid skewing model predictions.
        - Enhance feature engineering and explore additional models (e.g., XGBoost) for better accuracy.
        - Address dataset imbalance (if present) to improve dropout predictions, especially related to gender.
        """)

        st.subheader("Conclusions")
        st.write("""
        - Admission grades alone may not be a strong predictor of dropouts.
        - Gender has a notable influence on dropout rates, indicating a need for gender-targeted interventions.
        - The current model is a good baseline, but improvements can be made with additional data processing and model tuning.
        """)
 
    else:
        st.error("Failed to load data.")
 
if __name__ == '__main__':
    main()