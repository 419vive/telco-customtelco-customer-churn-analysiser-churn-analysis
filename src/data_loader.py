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
            print(f"❌ Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
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