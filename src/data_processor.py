import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self, delimiter=';'):
        """Load dataset from CSV file."""
        self.df = pd.read_csv(self.file_path, delimiter=delimiter)
        return self.df

    def explore_data(self):
        """Perform initial data exploration."""
        print(f"Dataset shape: {self.df.shape}")
        print("Data Types:\n", self.df.dtypes)
        return self.df.describe()

    def check_missing_values(self):
        """Check for missing values."""
        missing_values = self.df.isnull().sum()
        return missing_values[missing_values > 0]

    def visualize_missing_data(self):
        """Visualize missing data with a heatmap."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

    def clean_column_names(self):
        """Remove any tabs or whitespaces from column names."""
        self.df.columns = self.df.columns.str.replace('\t', '').str.strip()
        return self.df.columns

    def detect_outliers(self, num_cols):
        """Detect outliers using IQR."""
        outliers = {}
        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_in_col = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            outliers[col] = outliers_in_col
        return outliers

    def cap_outliers(self, col):
        """Cap outliers based on the 1st and 99th percentiles."""
        lower_cap = self.df[col].quantile(0.01)
        upper_cap = self.df[col].quantile(0.99)
        self.df[col] = self.df[col].clip(lower=lower_cap, upper=upper_cap)
        return self.df

    def encode_categorical(self):
        """Encode the 'Target' and other categorical columns."""
        self.df['Target'] = self.df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 2})
        return self.df['Target']

    def scale_features(self, num_cols):
        """Scale numerical columns using StandardScaler."""
        scaler = StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self.df[num_cols]

    def feature_engineering(self):
        """Create derived features."""
        # Age Group feature
        self.df['Age Group'] = pd.cut(self.df['Age at enrollment'], bins=[0, 18, 25, 35, 45, 100], labels=['<18', '18-25', '26-35', '36-45', '45+'])
        # Total curricular grade feature
        self.df['Total curricular grade'] = self.df['Curricular units 1st sem (grade)'] + self.df['Curricular units 2nd sem (grade)']
        # Full-time/Part-time feature
        self.df['Full-time/Part-time'] = self.df.apply(lambda x: 1 if (x['Curricular units 1st sem (enrolled)'] + x['Curricular units 2nd sem (enrolled)']) >= 6 else 0, axis=1)
        return self.df

    def descriptive_statistics(self):
        """Get descriptive statistics for both numerical and categorical columns."""
        numeric_stats = self.df.describe()
        categorical_stats = self.df.describe(include='object')
        return numeric_stats, categorical_stats

    def correlation_analysis(self):
        """Generate and visualize a correlation matrix."""
        numeric_df = self.df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()

    def t_test_admission_grade(self):
        """Perform t-test on Admission Grade (Dropout vs Graduate)."""
        dropouts = self.df[self.df['Target'] == 1]['Admission grade']
        graduates = self.df[self.df['Target'] == 0]['Admission grade']
        t_stat, p_value = stats.ttest_ind(dropouts, graduates)
        return t_stat, p_value

    def chi_square_test(self, col1, col2):
        """Perform chi-square test between two categorical columns."""
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2_stat, p_val

# Example Usage
if __name__ == '__main__':
    # Create an instance of DataProcessor
    processor = DataProcessor('../data/raw/data.csv')
    
    # Load the data
    processor.load_data()

    # Perform data exploration
    print("Data Exploration:\n", processor.explore_data())

    # Check for missing values
    print("Missing Values:\n", processor.check_missing_values())

    # Visualize missing data
    processor.visualize_missing_data()

    # Clean column names
    processor.clean_column_names()

    # Feature Engineering
    processor.feature_engineering()

    # Descriptive Statistics
    numeric_stats, categorical_stats = processor.descriptive_statistics()
    print("Numerical Stats:\n", numeric_stats)
    print("Categorical Stats:\n", categorical_stats)

    # Correlation Analysis
    processor.correlation_analysis()

    # Perform t-test and chi-square test
    t_stat, p_value = processor.t_test_admission_grade()
    print(f"T-test result (Dropout vs Graduate): T-stat={t_stat}, P-value={p_value}")

    chi2_stat, p_val = processor.chi_square_test('Gender', 'Target')
    print(f"Chi-square test (Gender vs Target): Chi2_stat={chi2_stat}, P-value={p_val}")