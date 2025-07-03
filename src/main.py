#!/usr/bin/env python3
"""
Main analysis script for Telco Customer Churn Analysis
Runs the complete CRISP-DM pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import TelcoDataLoader
from preprocessing import TelcoPreprocessor

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os

def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("🚀 TELCO CUSTOMER CHURN ANALYSIS")
    print("=" * 50)
    print("Starting complete CRISP-DM pipeline...\n")
    
    # Step 1: Business Understanding
    print("📊 PHASE 1: BUSINESS UNDERSTANDING")
    print("-" * 40)
    
    # Load data
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    if data is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Display business insights
    churn_rate = (data['Churn'] == 'Yes').mean() * 100
    total_customers = len(data)
    churned_customers = (data['Churn'] == 'Yes').sum()
    
    print(f"📈 Total Customers: {total_customers:,}")
    print(f"💔 Churned Customers: {churned_customers:,} ({churn_rate:.1f}%)")
    print(f"✅ Retained Customers: {total_customers - churned_customers:,} ({100-churn_rate:.1f}%)")
    
    # Revenue impact analysis
    avg_monthly_charge = data['MonthlyCharges'].mean()
    potential_monthly_loss = churned_customers * avg_monthly_charge
    print(f"💰 Average Monthly Charge: ${avg_monthly_charge:.2f}")
    print(f"💸 Potential Monthly Revenue Loss: ${potential_monthly_loss:,.2f}")
    print(f"💸 Potential Annual Revenue Loss: ${potential_monthly_loss * 12:,.2f}")
    
    # Step 2: Data Understanding
    print("\n🔍 PHASE 2: DATA UNDERSTANDING")
    print("-" * 40)
    
    # Data quality check
    missing_values = data.isnull().sum().sum()
    print(f"📊 Dataset Shape: {data.shape}")
    print(f"🔍 Missing Values: {missing_values}")
    print(f"📋 Columns: {len(data.columns)}")
    
    # Key insights
    contract_distribution = data['Contract'].value_counts(normalize=True) * 100
    print(f"\n📋 Contract Distribution:")
    for contract, percentage in contract_distribution.items():
        print(f"  {contract}: {percentage:.1f}%")
    
    # Step 3: Data Preparation
    print("\n🔧 PHASE 3: DATA PREPARATION")
    print("-" * 40)
    
    # Prepare data for modeling
    preprocessor = TelcoPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(data)
    
    print(f"✅ Training set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # Step 4: Modeling
    print("\n🤖 PHASE 4: MODELING")
    print("-" * 40)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = rf_model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
    
    # Step 5: Evaluation
    print("\n📊 PHASE 5: EVALUATION")
    print("-" * 40)
    
    # Classification report
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 TOP 10 FEATURE IMPORTANCE:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Step 6: Deployment/Recommendations
    print("\n🚀 PHASE 6: DEPLOYMENT & RECOMMENDATIONS")
    print("-" * 40)
    
    # Save model
    if not os.path.exists('../models'):
        os.makedirs('../models')
    
    model_path = '../models/telco_churn_model.pkl'
    joblib.dump(rf_model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save preprocessor
    preprocessor_path = '../models/preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✅ Preprocessor saved to: {preprocessor_path}")
    
    # Business recommendations
    print("\n💼 BUSINESS RECOMMENDATIONS:")
    print("1. 🎯 Focus on customers with month-to-month contracts")
    print("2. 💰 Monitor high monthly charges as risk indicator")
    print("3. ⏰ Implement early intervention for new customers")
    print("4. 🔧 Use model predictions for targeted retention campaigns")
    print("5. 📊 Monitor model performance monthly")
    
    # Expected business impact
    print("\n🏆 EXPECTED BUSINESS IMPACT:")
    print(f"• Model can identify {accuracy:.1%} of churn cases correctly")
    print(f"• Potential to reduce churn by 5-10% with targeted interventions")
    print(f"• Annual revenue protection: ${potential_monthly_loss * 12 * 0.05:,.2f} (5% reduction)")
    print(f"• Improved marketing ROI through focused retention efforts")
    
    # Create visualizations
    create_analysis_visualizations(data, y_test, y_pred, y_pred_proba, feature_importance_df)
    
    print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

def create_analysis_visualizations(data, y_test, y_pred, y_pred_proba, feature_importance_df):
    """
    Create comprehensive visualizations for the analysis
    """
    print("\n📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a comprehensive visualization dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Churn Distribution
    plt.subplot(3, 3, 1)
    churn_counts = data['Churn'].value_counts()
    plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Customer Churn Distribution', fontweight='bold')
    
    # 2. Monthly Charges vs Churn
    plt.subplot(3, 3, 2)
    sns.boxplot(data=data, x='Churn', y='MonthlyCharges')
    plt.title('Monthly Charges vs Churn', fontweight='bold')
    plt.xlabel('Churn Status')
    plt.ylabel('Monthly Charges ($)')
    
    # 3. Tenure vs Churn
    plt.subplot(3, 3, 3)
    sns.histplot(data=data, x='tenure', hue='Churn', bins=30, alpha=0.7)
    plt.title('Customer Tenure vs Churn', fontweight='bold')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Count')
    
    # 4. Contract Type vs Churn
    plt.subplot(3, 3, 4)
    contract_churn = pd.crosstab(data['Contract'], data['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar')
    plt.title('Churn Rate by Contract Type', fontweight='bold')
    plt.xlabel('Contract Type')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Churn Status')
    plt.xticks(rotation=45)
    
    # 5. Confusion Matrix
    plt.subplot(3, 3, 5)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 6. ROC Curve
    plt.subplot(3, 3, 6)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    # 7. Feature Importance
    plt.subplot(3, 3, 7)
    top_features = feature_importance_df.head(10)
    bars = plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance', fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontweight='bold')
    
    # 8. Payment Method vs Churn
    plt.subplot(3, 3, 8)
    payment_churn = pd.crosstab(data['PaymentMethod'], data['Churn'], normalize='index') * 100
    payment_churn.plot(kind='bar')
    plt.title('Churn Rate by Payment Method', fontweight='bold')
    plt.xlabel('Payment Method')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Churn Status')
    plt.xticks(rotation=45)
    
    # 9. Internet Service vs Churn
    plt.subplot(3, 3, 9)
    internet_churn = pd.crosstab(data['InternetService'], data['Churn'], normalize='index') * 100
    internet_churn.plot(kind='bar')
    plt.title('Churn Rate by Internet Service', fontweight='bold')
    plt.xlabel('Internet Service')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Churn Status')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the visualization
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    plt.savefig('../results/telco_churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved to: ../results/telco_churn_analysis_dashboard.png")

if __name__ == "__main__":
    main() 