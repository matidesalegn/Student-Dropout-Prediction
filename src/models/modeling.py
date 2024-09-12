# src/models/modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_cleaned_data(file_path):
    """
    Load the cleaned dataset from a CSV file.

    Parameters:
    - file_path (str): Path to the cleaned CSV file.

    Returns:
    - df (DataFrame): Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
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

def split_data(df, target_col):
    """
    Split the data into training and testing sets.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - target_col (str): Name of the target column.

    Returns:
    - X_train, X_test, y_train, y_test: Data splits for training and testing.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical features if any
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_and_evaluate_models(X_train, X_test, y_train, y_test, target_names):
    """
    Build and evaluate several models.

    Parameters:
    - X_train, X_test, y_train, y_test: Data splits for training and testing.
    - target_names (list): List of target class names as strings.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Classification Report': classification_report(y_test, y_pred, target_names=target_names)
        }
        
        print(f"\n{name} Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Save the model
        model_path = f'../../models/{name.replace(" ", "_")}.joblib'
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    return results

def main():
    # Path to the cleaned data
    data_path = '../../data/processed/cleaned_data.csv'
    
    # Load data
    df = load_cleaned_data(data_path)
    
    # Create Age Group feature
    df = create_age_group(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, 'Target')
    
    # Scale features (important for models like Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get target class names as strings
    target_names = [str(cls) for cls in df['Target'].unique()]
    
    # Build and evaluate models
    results = build_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, target_names)
    
    return results

if __name__ == '__main__':
    main()
