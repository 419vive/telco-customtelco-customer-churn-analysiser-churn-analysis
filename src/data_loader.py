"""
Data loading utilities for Telco Customer Churn Analysis
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TelcoDataLoader:
    """
    Data loader class for Telco Customer Churn dataset
    """
    
    def __init__(self, data_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file
        """
        self.data_path = data_path
        self.data = None
        
    def generate_synthetic_data(self, n_samples=7043):
        """
        Generate synthetic telco churn data matching the original dataset structure
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        pandas.DataFrame
            Synthetic dataset
        """
        print("🔄 Generating synthetic telco churn data...")
        np.random.seed(42)
        
        # Generate synthetic data
        data = {
            'customerID': [f'ID{str(i).zfill(6)}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        }
        
        # Generate tenure (1-72 months) with realistic distribution
        data['tenure'] = np.random.gamma(shape=2, scale=15, size=n_samples)
        data['tenure'] = np.clip(data['tenure'], 1, 72).astype(int)
        
        # Services
        data['PhoneService'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
        data['MultipleLines'] = np.where(
            data['PhoneService'] == 'Yes',
            np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
            'No phone service'
        )
        
        data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
        
        # Internet-dependent services
        for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']:
            data[service] = np.where(
                data['InternetService'] == 'No',
                'No internet service',
                np.random.choice(['Yes', 'No'], n_samples, p=[0.44, 0.56])
            )
        
        # Contract and billing
        data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                          n_samples, p=[0.55, 0.21, 0.24])
        data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
        data['PaymentMethod'] = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            n_samples, p=[0.33, 0.23, 0.22, 0.22]
        )
        
        # Generate monthly charges based on services
        base_charge = 20
        data['MonthlyCharges'] = base_charge
        
        # Add charges for services
        data['MonthlyCharges'] += (data['PhoneService'] == 'Yes') * np.random.normal(10, 2, n_samples)
        data['MonthlyCharges'] += (data['MultipleLines'] == 'Yes') * np.random.normal(15, 2, n_samples)
        data['MonthlyCharges'] += (data['InternetService'] == 'DSL') * np.random.normal(30, 5, n_samples)
        data['MonthlyCharges'] += (data['InternetService'] == 'Fiber optic') * np.random.normal(50, 5, n_samples)
        
        # Add charges for additional services
        for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']:
            data['MonthlyCharges'] += (data[service] == 'Yes') * np.random.normal(10, 2, n_samples)
        
        data['MonthlyCharges'] = np.clip(data['MonthlyCharges'], 18.25, 118.75)
        
        # Calculate total charges based on tenure and monthly charges
        data['TotalCharges'] = data['tenure'] * data['MonthlyCharges'] * np.random.uniform(0.95, 1.05, n_samples)
        
        # Generate churn based on realistic patterns
        churn_probability = np.zeros(n_samples)
        
        # Factors increasing churn probability
        churn_probability += (data['Contract'] == 'Month-to-month') * 0.3
        churn_probability += (data['tenure'] < 12) * 0.2
        churn_probability += (data['InternetService'] == 'Fiber optic') * 0.1
        churn_probability += (data['PaymentMethod'] == 'Electronic check') * 0.15
        churn_probability += (data['MonthlyCharges'] > 70) * 0.1
        churn_probability += (data['SeniorCitizen'] == 1) * 0.1
        churn_probability += (data['Partner'] == 'No') * 0.05
        churn_probability += (data['OnlineSecurity'] == 'No') * 0.05
        churn_probability += (data['TechSupport'] == 'No') * 0.05
        
        # Factors decreasing churn probability
        churn_probability -= (data['Contract'] == 'Two year') * 0.3
        churn_probability -= (data['tenure'] > 48) * 0.2
        churn_probability -= (data['Dependents'] == 'Yes') * 0.1
        
        # Ensure probability is between 0 and 1
        churn_probability = np.clip(churn_probability, 0, 0.8)
        
        # Generate churn (targeting ~26.5% churn rate)
        data['Churn'] = np.where(np.random.random(n_samples) < churn_probability, 'Yes', 'No')
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert TotalCharges to string and add some blanks (as in original dataset)
        df.loc[df['tenure'] == 0, 'TotalCharges'] = ' '
        df['TotalCharges'] = df['TotalCharges'].astype(str)
        
        print(f"✅ Generated synthetic dataset with {n_samples} samples")
        return df
        
    def load_data(self):
        """
        Load the Telco Customer Churn dataset
        
        Returns:
        --------
        pandas.DataFrame
            Loaded dataset
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.data.shape}")
            print(f"📁 File: {self.data_path}")
            return self.data
        except FileNotFoundError:
            print(f"⚠️  Warning: File not found at {self.data_path}")
            print("📊 Generating synthetic data for demonstration...")
            self.data = self.generate_synthetic_data()
            return self.data
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print("📊 Generating synthetic data for demonstration...")
            self.data = self.generate_synthetic_data()
            return self.data
    
    def get_data_info(self):
        """
        Get basic information about the dataset
        
        Returns:
        --------
        dict
            Dictionary containing dataset information
        """
        if self.data is None:
            print("❌ No data loaded. Call load_data() first.")
            return None
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'churn_distribution': self.data['Churn'].value_counts().to_dict() if 'Churn' in self.data.columns else None
        }
        
        return info
    
    def display_data_info(self):
        """
        Display comprehensive dataset information
        """
        if self.data is None:
            print("❌ No data loaded. Call load_data() first.")
            return
            
        print("=" * 60)
        print("📊 TELCO CUSTOMER CHURN DATASET INFORMATION")
        print("=" * 60)
        
        # Basic info
        print(f"📈 Dataset Shape: {self.data.shape}")
        print(f"💾 Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        # Column information
        print("📋 COLUMNS:")
        for i, col in enumerate(self.data.columns, 1):
            dtype = str(self.data[col].dtype)
            missing = self.data[col].isnull().sum()
            unique = self.data[col].nunique()
            print(f"  {i:2d}. {col:<25} | {dtype:<10} | Missing: {missing:3d} | Unique: {unique:3d}")
        
        print()
        
        # Missing values
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            print("⚠️  MISSING VALUES:")
            for col, missing in missing_data[missing_data > 0].items():
                percentage = (missing / len(self.data)) * 100
                print(f"  {col}: {missing} ({percentage:.1f}%)")
        else:
            print("✅ No missing values found!")
        
        print()
        
        # Churn distribution
        if 'Churn' in self.data.columns:
            print("🎯 CHURN DISTRIBUTION:")
            churn_counts = self.data['Churn'].value_counts()
            churn_percentages = self.data['Churn'].value_counts(normalize=True) * 100
            
            for status, count in churn_counts.items():
                percentage = churn_percentages[status]
                print(f"  {status}: {count:,} ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def get_categorical_columns(self):
        """
        Get list of categorical columns
        
        Returns:
        --------
        list
            List of categorical column names
        """
        if self.data is None:
            return []
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        return categorical_cols
    
    def get_numerical_columns(self):
        """
        Get list of numerical columns
        
        Returns:
        --------
        list
            List of numerical column names
        """
        if self.data is None:
            return []
        
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numerical_cols
    
    def sample_data(self, n=5):
        """
        Display a sample of the data
        
        Parameters:
        -----------
        n : int
            Number of rows to display
        """
        if self.data is None:
            print("❌ No data loaded. Call load_data() first.")
            return
        
        print(f"📋 SAMPLE DATA (First {n} rows):")
        print("=" * 80)
        print(self.data.head(n).to_string())
        print("=" * 80)

def load_telco_data(data_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Convenience function to load Telco data
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    loader = TelcoDataLoader(data_path)
    return loader.load_data()

if __name__ == "__main__":
    # Test the data loader
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    if data is not None:
        loader.display_data_info()
        loader.sample_data(3) 