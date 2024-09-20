import streamlit as st
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  # For loading saved models

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction App",  # Page title
    page_icon="📊",  # Page icon (emoji or file)
    layout="wide",  # Wide layout for the app
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

# Sidebar for color theme and settings
st.sidebar.title("Visualization Settings")
selected_color_theme = st.sidebar.selectbox(
    "Choose color theme for visualizations:",
    ["viridis", "inferno", "plasma", "magma"]
)
st.sidebar.write(f"Selected Color Theme: {selected_color_theme}")

# Add src to the path to allow relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import data processing modules (assumed to be present in the given paths)
from data.load_data import load_dataset
from data.clean_data import clean_column_names, handle_missing_values, detect_outliers, cap_outliers
from data.transform_data import encode_categorical_variables, scale_features, feature_engineering

# Define a function for univariate analysis
def univariate_analysis(df):
    st.subheader("Univariate Analysis")
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

    st.write("### Bar Charts for Categorical Variables")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"**{col}**")
        st.bar_chart(df[col].value_counts())

# Define a function for bivariate analysis
def bivariate_analysis(df):
    st.subheader("Bivariate Analysis")
    st.write("### Correlation Matrix")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap=selected_color_theme, ax=ax)
    st.pyplot(fig)

    st.write("### Scatter Plot Pairs (Limited Features)")
    selected_columns = numerical_columns[:5]
    pair_plot = sns.pairplot(df[selected_columns], diag_kind="kde", palette=selected_color_theme)
    st.pyplot(pair_plot)

# Define a function for multivariate analysis (PCA)
def multivariate_analysis(df, selected_color_theme):
    st.subheader("Multivariate Analysis")
    st.write("### Principal Component Analysis (PCA)")

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_scaled = StandardScaler().fit_transform(df[numerical_columns])

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)

    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df['Target_Code'] = df['Target'].map(target_mapping)

    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Target_Code'] = df['Target_Code']

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Target_Code'], cmap=selected_color_theme)
    legend_labels = {v: k for k, v in target_mapping.items()}
    handles, _ = scatter.legend_elements()
    legend = ax.legend(handles, [legend_labels[i] for i in range(len(handles))], title="Target")
    ax.add_artist(legend)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Dataset')
    st.pyplot(fig)

# Load the XGBoost model
def load_xgboost_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))
    xgboost_model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    return joblib.load(xgboost_model_path)

# Manual mapping for numeric predictions to human-readable labels
prediction_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

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
        df = clean_column_names(df)
        df = handle_missing_values(df)

        feature_cols = df.drop(columns=['Target']).columns.tolist() if 'Target' in df.columns else df.columns.tolist()

        if st.checkbox("Show raw data"):
            st.write(df.head())

        if st.checkbox("Perform Univariate Analysis"):
            univariate_analysis(df)

        if st.checkbox("Perform Bivariate Analysis"):
            bivariate_analysis(df)

        if st.checkbox("Perform Multivariate Analysis"):
            multivariate_analysis(df, selected_color_theme)

        if st.checkbox("Show initial data exploration"):
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write(df.dtypes)
            st.write(df.describe())

        if st.checkbox("Check for missing values"):
            missing_values = df.isnull().sum()
            st.write(missing_values[missing_values > 0])
            if missing_values.sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, cmap=selected_color_theme)
                st.pyplot(plt.gcf())

        num_cols = ['Admission grade', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
        outliers = detect_outliers(df, num_cols)
        if st.checkbox("Show outliers"):
            for col, outliers_col in outliers.items():
                st.write(f"Outliers in {col}:")
                st.write(outliers_col)

        if st.checkbox("Show box plots for numerical columns"):
            fig, axes = plt.subplots(1, len(num_cols), figsize=(12, 8))
            for i, col in enumerate(num_cols):
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(col)
            st.pyplot(fig)

        df = cap_outliers(df, num_cols)
        df = encode_categorical_variables(df)
        df = feature_engineering(df)
        scale_cols = ['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
        df = scale_features(df, scale_cols)

        if st.checkbox("Show preprocessed data"):
            st.write(df.head())

        X = df.drop(columns='Target')
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.checkbox("Show feature importance"):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            st.write(feature_importance_df)

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
            st.pyplot(fig)

        st.write("### Model Prediction")
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

        with st.form(key='prediction_form'):
            st.write("Enter details for prediction:")
            user_input = {}
            for col in categorical_cols:
                user_input[col] = st.selectbox(f"{col}", df[col].unique())
            for col in numerical_cols:
                user_input[col] = st.number_input(f"{col}")
            submit_button = st.form_submit_button("Predict")

            if submit_button:
                input_data = pd.DataFrame([user_input])
                input_data = encode_categorical_variables(input_data)
                input_data = feature_engineering(input_data)
                input_data = scale_features(input_data, numerical_cols)
                input_data = input_data.reindex(columns=feature_cols, fill_value=0)
                numeric_prediction = xgboost_model.predict(input_data)[0]
                label_prediction = prediction_mapping.get(numeric_prediction, "Unknown")
                st.success(f"Prediction: {label_prediction}")

        st.write("### Insights, Recommendations, and Conclusions")
        st.subheader("Insights")
        st.write("""
        - The dataset contains 4,424 rows and 38 columns.
        - Outliers were detected in key numerical variables such as Admission grade and semester grades.
        - Gender shows a significant impact on dropout rates according to the Chi-square test.
        """)

        st.subheader("Recommendations")
        st.write("""
        - Investigate missing values and handle them appropriately to improve model performance.
        - Consider outlier treatment techniques like transformations to avoid skewing model predictions.
        """)

        st.subheader("Conclusions")
        st.write("""
        - Admission grades alone may not be a strong predictor of dropouts.
        - Gender has a notable influence on dropout rates, indicating a need for gender-targeted interventions.
        """)

    else:
        st.error("Failed to load data.")

if __name__ == '__main__':
    main()