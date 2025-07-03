"""
Data preprocessing utilities for Telco Customer Churn Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TelcoPreprocessor:
    """
    Data preprocessing class for Telco Customer Churn dataset
    """
    
    def __init__(self):
        """
        Initialize the preprocessor
        """
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def clean_data(self, data):
        """
        Clean the raw data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw dataset
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataset
        """
        print("🧹 Starting data cleaning...")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # 1. Handle TotalCharges - convert to numeric and fill missing values
        print("  📊 Converting TotalCharges to numeric...")
        cleaned_data['TotalCharges'] = pd.to_numeric(cleaned_data['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (for new customers)
        missing_charges = cleaned_data['TotalCharges'].isnull().sum()
        if missing_charges > 0:
            print(f"  ⚠️  Found {missing_charges} missing values in TotalCharges")
            cleaned_data['TotalCharges'].fillna(0, inplace=True)
        
        # 2. Create additional features
        print("  🔧 Creating additional features...")
        
        # Total charges per month
        cleaned_data['ChargesPerMonth'] = cleaned_data['TotalCharges'] / cleaned_data['tenure'].replace(0, 1)
        
        # Contract duration (in months)
        contract_duration = {
            'Month-to-month': 1,
            'One year': 12,
            'Two year': 24
        }
        cleaned_data['ContractDuration'] = cleaned_data['Contract'].map(contract_duration)
        
        # Service count
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        cleaned_data['ServiceCount'] = 0
        for col in service_cols:
            cleaned_data['ServiceCount'] += (cleaned_data[col] != 'No').astype(int)
        
        # Internet service type
        cleaned_data['HasInternet'] = (cleaned_data['InternetService'] != 'No').astype(int)
        
        # Payment method type
        cleaned_data['ElectronicPayment'] = cleaned_data['PaymentMethod'].str.contains('Electronic').astype(int)
        
        # 3. Remove customerID (not useful for modeling)
        if 'customerID' in cleaned_data.columns:
            cleaned_data.drop('customerID', axis=1, inplace=True)
            print("  🗑️  Removed customerID column")
        
        print("✅ Data cleaning completed!")
        return cleaned_data
    
    def encode_categorical_variables(self, data, fit=True):
        """
        Encode categorical variables using Label Encoding
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset with categorical variables
        fit : bool
            Whether to fit new encoders or use existing ones
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with encoded categorical variables
        """
        print("🔤 Encoding categorical variables...")
        
        # Get categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        encoded_data = data.copy()
        
        for col in categorical_cols:
            if fit:
                # Create new encoder
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col])
                self.label_encoders[col] = le
                print(f"  ✅ Encoded {col}: {list(le.classes_)}")
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    encoded_data[col] = encoded_data[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                    print(f"  ✅ Applied existing encoder for {col}")
                else:
                    print(f"  ⚠️  No encoder found for {col}, skipping...")
        
        return encoded_data
    
    def scale_numerical_features(self, data, fit=True):
        """
        Scale numerical features using StandardScaler
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset with numerical variables
        fit : bool
            Whether to fit new scaler or use existing one
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with scaled numerical variables
        """
        print("📏 Scaling numerical features...")
        
        # Get numerical columns (excluding target variable)
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        scaled_data = data.copy()
        
        if fit:
            # Fit and transform
            scaled_data[numerical_cols] = self.scaler.fit_transform(scaled_data[numerical_cols])
            print(f"  ✅ Scaled {len(numerical_cols)} numerical features")
        else:
            # Transform only
            scaled_data[numerical_cols] = self.scaler.transform(scaled_data[numerical_cols])
            print(f"  ✅ Applied existing scaler to {len(numerical_cols)} features")
        
        return scaled_data
    
    def prepare_data(self, data, target_col='Churn', test_size=0.2, random_state=42):
        """
        Complete data preparation pipeline
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw dataset
        target_col : str
            Name of target variable
        test_size : float
            Proportion of test set
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, feature_names)
        """
        print("🚀 Starting complete data preparation pipeline...")
        
        # 1. Clean data
        cleaned_data = self.clean_data(data)
        
        # 2. Encode categorical variables
        encoded_data = self.encode_categorical_variables(cleaned_data, fit=True)
        
        # 3. Scale numerical features
        scaled_data = self.scale_numerical_features(encoded_data, fit=True)
        
        # 4. Separate features and target
        X = scaled_data.drop(target_col, axis=1)
        y = scaled_data[target_col]
        
        # 5. Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.is_fitted = True
        
        print("✅ Data preparation completed!")
        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Test set: {X_test.shape}")
        print(f"🎯 Target distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"🎯 Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def transform_new_data(self, data):
        """
        Transform new data using fitted preprocessors
        
        Parameters:
        -----------
        data : pandas.DataFrame
            New dataset to transform
            
        Returns:
        --------
        pandas.DataFrame
            Transformed dataset
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        print("🔄 Transforming new data...")
        
        # 1. Clean data
        cleaned_data = self.clean_data(data)
        
        # 2. Encode categorical variables (using existing encoders)
        encoded_data = self.encode_categorical_variables(cleaned_data, fit=False)
        
        # 3. Scale numerical features (using existing scaler)
        scaled_data = self.scale_numerical_features(encoded_data, fit=False)
        
        print("✅ New data transformation completed!")
        return scaled_data
    
    def get_feature_importance_ranking(self, feature_names):
        """
        Get feature importance ranking based on domain knowledge
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        dict
            Feature importance scores
        """
        # Define feature importance based on business knowledge
        importance_scores = {
            'tenure': 0.95,
            'Contract': 0.90,
            'MonthlyCharges': 0.85,
            'TotalCharges': 0.80,
            'PaymentMethod': 0.75,
            'InternetService': 0.70,
            'OnlineSecurity': 0.65,
            'TechSupport': 0.65,
            'ContractDuration': 0.60,
            'ServiceCount': 0.55,
            'PaperlessBilling': 0.50,
            'ChargesPerMonth': 0.45,
            'HasInternet': 0.40,
            'ElectronicPayment': 0.35,
            'Partner': 0.30,
            'Dependents': 0.30,
            'SeniorCitizen': 0.25,
            'gender': 0.20
        }
        
        # Create ranking for available features
        feature_ranking = {}
        for feature in feature_names:
            if feature in importance_scores:
                feature_ranking[feature] = importance_scores[feature]
            else:
                feature_ranking[feature] = 0.10  # Default low importance
        
        # Sort by importance
        feature_ranking = dict(sorted(feature_ranking.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return feature_ranking

def create_feature_importance_plot(feature_ranking, top_n=15):
    """
    Create feature importance visualization
    
    Parameters:
    -----------
    feature_ranking : dict
        Feature importance scores
    top_n : int
        Number of top features to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top N features
    top_features = dict(list(feature_ranking.items())[:top_n])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    features = list(top_features.keys())
    scores = list(top_features.values())
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(features)), scores)
    
    # Customize plot
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top {top_n} Feature Importance (Domain Knowledge)', fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import TelcoDataLoader
    
    # Load data
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    if data is not None:
        # Initialize preprocessor
        preprocessor = TelcoPreprocessor()
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(data)
        
        # Get feature importance
        feature_ranking = preprocessor.get_feature_importance_ranking(feature_names)
        
        print("\\n🏆 FEATURE IMPORTANCE RANKING:")
        print("=" * 50)
        for i, (feature, score) in enumerate(feature_ranking.items(), 1):
            print(f"{i:2d}. {feature:<20}: {score:.2f}") 