#!/usr/bin/env python3
"""
Customer Segmentation Visualization for Retention Strategy
Shows which customers to keep vs let go
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_loader import TelcoDataLoader

def main():
    print("🎯 CUSTOMER SEGMENTATION: WHO TO KEEP VS WHO TO LET GO")
    print("=" * 60)
    
    # Load data
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    # Create customer segments based on value and risk
    def create_simple_segments(data):
        # Value based on monthly charges
        data['HighValue'] = data['MonthlyCharges'] > data['MonthlyCharges'].median()
        
        # Risk based on contract type
        data['HighRisk'] = data['Contract'] == 'Month-to-month'
        
        # Create segments
        conditions = [
            (data['HighValue'] & ~data['HighRisk']),  # High Value, Low Risk
            (data['HighValue'] & data['HighRisk']),   # High Value, High Risk
            (~data['HighValue'] & ~data['HighRisk']), # Low Value, Low Risk
            (~data['HighValue'] & data['HighRisk'])   # Low Value, High Risk
        ]
        choices = ['VIP - Keep', 'High Value High Risk - Fight', 'Low Value Low Risk - Let Go', 'Low Value High Risk - Let Go']
        data['Segment'] = np.select(conditions, choices, default='Other')
        
        return data
    
    data = create_simple_segments(data)
    
    # Analyze segments
    segment_analysis = data.groupby('Segment').agg({
        'customerID': 'count',
        'MonthlyCharges': ['sum', 'mean'],
        'Churn': lambda x: (x == 'Yes').mean() * 100
    }).round(2)
    
    segment_analysis.columns = ['Count', 'Total_Revenue', 'Avg_Revenue', 'Churn_Rate']
    
    print("\n📊 SEGMENT ANALYSIS:")
    print(segment_analysis.to_string())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Segment Distribution
    segment_counts = data['Segment'].value_counts()
    colors = ['green', 'orange', 'lightblue', 'red']
    axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Customer Segment Distribution', fontweight='bold', fontsize=14)
    
    # 2. Churn Rate by Segment
    churn_by_segment = data.groupby('Segment')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    bars = axes[0, 1].bar(churn_by_segment.index, churn_by_segment.values, color=colors)
    axes[0, 1].set_title('Churn Rate by Segment', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Churn Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, churn_by_segment.values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Monthly Revenue by Segment
    revenue_by_segment = data.groupby('Segment')['MonthlyCharges'].sum()
    bars = axes[1, 0].bar(revenue_by_segment.index, revenue_by_segment.values, color=colors)
    axes[1, 0].set_title('Monthly Revenue by Segment', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('Monthly Revenue ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, revenue_by_segment.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                       f'${value:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 4. Value vs Risk Scatter
    for segment in data['Segment'].unique():
        segment_data = data[data['Segment'] == segment]
        color_map = {
            'VIP - Keep': 'green',
            'High Value High Risk - Fight': 'orange',
            'Low Value Low Risk - Let Go': 'lightblue',
            'Low Value High Risk - Let Go': 'red'
        }
        axes[1, 1].scatter(segment_data['MonthlyCharges'], 
                          segment_data['tenure'], 
                          c=color_map[segment], 
                          label=segment, 
                          alpha=0.6, s=30)
    
    axes[1, 1].set_xlabel('Monthly Charges ($)')
    axes[1, 1].set_ylabel('Tenure (months)')
    axes[1, 1].set_title('Customer Value vs Tenure by Segment', fontweight='bold', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    print("\n🎯 RETENTION STRATEGY RECOMMENDATIONS:")
    print("=" * 60)
    
    recommendations = {
        'VIP - Keep': {
            'action': 'VIP Treatment',
            'priority': 'Highest',
            'strategy': 'Exclusive benefits, dedicated support',
            'budget': 'High'
        },
        'High Value High Risk - Fight': {
            'action': 'Aggressive Retention',
            'priority': 'Critical',
            'strategy': 'Immediate intervention, personalized offers',
            'budget': 'High'
        },
        'Low Value Low Risk - Let Go': {
            'action': 'Let Go',
            'priority': 'Low',
            'strategy': 'Minimal effort, basic service',
            'budget': 'Minimal'
        },
        'Low Value High Risk - Let Go': {
            'action': 'Let Go',
            'priority': 'Lowest',
            'strategy': 'No retention effort',
            'budget': 'None'
        }
    }
    
    for segment, details in recommendations.items():
        if segment in segment_analysis.index:
            count = segment_analysis.loc[segment, 'Count']
            revenue = segment_analysis.loc[segment, 'Total_Revenue']
            churn_rate = segment_analysis.loc[segment, 'Churn_Rate']
            
            print(f"\n🎯 {segment}")
            print(f"   📊 Count: {count:,} customers")
            print(f"   💰 Monthly Revenue: ${revenue:,.2f}")
            print(f"   📉 Churn Rate: {churn_rate:.1f}%")
            print(f"   🎯 Action: {details['action']}")
            print(f"   ⚡ Priority: {details['priority']}")
            print(f"   💡 Strategy: {details['strategy']}")
            print(f"   💸 Budget: {details['budget']}")
    
    print("\n🔥 FINAL RECOMMENDATION:")
    print("=" * 40)
    print("🎯 FOCUS ON: VIP customers and High Value High Risk customers")
    print("💡 STRATEGY: Aggressive retention for high-value at-risk customers")
    print("💰 BUDGET: Allocate 80% of retention budget to top 2 segments")
    print("🚫 LET GO: Low value customers (save resources for high-value retention)")

if __name__ == "__main__":
    main() 