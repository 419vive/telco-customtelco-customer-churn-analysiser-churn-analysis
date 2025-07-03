#!/usr/bin/env python3
"""
Abstract Visualizations for Customer Segmentation and Retention Strategy
Creates compelling visual representations of our analysis and strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from data_loader import TelcoDataLoader

def create_abstract_customer_segmentation():
    """Create abstract customer segmentation visualization"""
    
    # Load data
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    # Create segments
    def create_simple_segments(data):
        data['HighValue'] = data['MonthlyCharges'] > data['MonthlyCharges'].median()
        data['HighRisk'] = data['Contract'] == 'Month-to-month'
        
        conditions = [
            (data['HighValue'] & ~data['HighRisk']),
            (data['HighValue'] & data['HighRisk']),
            (~data['HighValue'] & ~data['HighRisk']),
            (~data['HighValue'] & data['HighRisk'])
        ]
        choices = ['VIP - Keep', 'High Value High Risk - Fight', 'Low Value Low Risk - Let Go', 'Low Value High Risk - Let Go']
        data['Segment'] = np.select(conditions, choices, default='Other')
        return data
    
    data = create_simple_segments(data)
    
    # Create abstract visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Abstract Customer Universe
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Create abstract customer clusters
    colors = ['gold', 'red', 'lightblue', 'darkblue']
    segments = ['VIP - Keep', 'High Value High Risk - Fight', 'Low Value Low Risk - Let Go', 'Low Value High Risk - Let Go']
    
    # VIP - Keep (gold stars)
    for i in range(20):
        x = np.random.uniform(1, 3)
        y = np.random.uniform(7, 9)
        ax1.scatter(x, y, s=100, c='gold', marker='*', alpha=0.8)
    
    # High Value High Risk (red circles)
    for i in range(30):
        x = np.random.uniform(6, 9)
        y = np.random.uniform(6, 9)
        ax1.scatter(x, y, s=80, c='red', alpha=0.7)
    
    # Low Value Low Risk (light blue squares)
    for i in range(25):
        x = np.random.uniform(1, 4)
        y = np.random.uniform(1, 4)
        ax1.scatter(x, y, s=60, c='lightblue', marker='s', alpha=0.6)
    
    # Low Value High Risk (dark blue triangles)
    for i in range(25):
        x = np.random.uniform(6, 9)
        y = np.random.uniform(1, 4)
        ax1.scatter(x, y, s=70, c='darkblue', marker='^', alpha=0.6)
    
    ax1.set_title('Customer Universe: Value vs Risk Segmentation', fontsize=16, fontweight='bold')
    ax1.text(5, 9.5, 'HIGH VALUE', ha='center', fontsize=12, fontweight='bold')
    ax1.text(5, 0.5, 'LOW VALUE', ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 5, 'LOW RISK', rotation=90, fontsize=12, fontweight='bold')
    ax1.text(9.5, 5, 'HIGH RISK', rotation=90, fontsize=12, fontweight='bold')
    
    # 2. Revenue Flow Visualization
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Create revenue flow diagram
    segment_analysis = data.groupby('Segment').agg({
        'customerID': 'count',
        'MonthlyCharges': 'sum'
    })
    
    # VIP segment (large circle)
    vip_circle = Circle((3, 7), 1.5, color='gold', alpha=0.8)
    ax2.add_patch(vip_circle)
    ax2.text(3, 7, 'VIP\n$139K', ha='center', va='center', fontweight='bold')
    
    # High Value High Risk (large circle with warning)
    hvhr_circle = Circle((7, 7), 1.8, color='red', alpha=0.7)
    ax2.add_patch(hvhr_circle)
    ax2.text(7, 7, 'HIGH VALUE\nHIGH RISK\n$180K', ha='center', va='center', fontweight='bold', color='white')
    
    # Low Value segments (smaller circles)
    lvlr_circle = Circle((2, 3), 0.8, color='lightblue', alpha=0.6)
    ax2.add_patch(lvlr_circle)
    ax2.text(2, 3, 'LOW VALUE\n$60K', ha='center', va='center', fontsize=8)
    
    lvhr_circle = Circle((8, 3), 0.8, color='darkblue', alpha=0.6)
    ax2.add_patch(lvhr_circle)
    ax2.text(8, 3, 'LOW VALUE\n$78K', ha='center', va='center', fontsize=8)
    
    # Revenue flow arrows
    ax2.arrow(3, 5.5, 0, -1, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    ax2.arrow(7, 5.2, 0, -1, head_width=0.2, head_length=0.2, fc='orange', ec='orange', linewidth=3)
    ax2.arrow(2, 1.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='gray', ec='gray', linewidth=2)
    ax2.arrow(8, 1.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='gray', ec='gray', linewidth=2)
    
    ax2.set_title('Revenue Flow: Monthly Revenue by Segment', fontsize=16, fontweight='bold')
    
    # 3. Churn Risk Heatmap
    ax3 = axes[1, 0]
    
    # Create churn risk matrix
    churn_data = np.array([
        [10.7, 52.8],  # VIP, High Value High Risk
        [3.3, 31.4]    # Low Value Low Risk, Low Value High Risk
    ])
    
    im = ax3.imshow(churn_data, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Low Risk', 'High Risk'])
    ax3.set_yticklabels(['High Value', 'Low Value'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, f'{churn_data[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax3.set_title('Churn Risk Heatmap (%)', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Churn Rate (%)')
    
    # 4. Strategy Action Plan
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Strategy boxes
    # VIP Strategy
    vip_box = Rectangle((0.5, 7), 4, 2, color='gold', alpha=0.8)
    ax4.add_patch(vip_box)
    ax4.text(2.5, 8, 'VIP STRATEGY\nENHANCE EXPERIENCE', ha='center', va='center', fontweight='bold')
    
    # High Value High Risk Strategy
    hvhr_box = Rectangle((5.5, 7), 4, 2, color='red', alpha=0.7)
    ax4.add_patch(hvhr_box)
    ax4.text(7.5, 8, 'AGGRESSIVE RETENTION\nFIGHT TO KEEP', ha='center', va='center', fontweight='bold', color='white')
    
    # Low Value Strategy
    lv_box = Rectangle((2.5, 4), 5, 2, color='lightgray', alpha=0.6)
    ax4.add_patch(lv_box)
    ax4.text(5, 5, 'LET GO NATURALLY\nMINIMAL EFFORT', ha='center', va='center', fontweight='bold')
    
    # Budget allocation
    ax4.text(2.5, 2, 'BUDGET: $80K → Top 2 Segments', ha='center', fontsize=12, fontweight='bold')
    ax4.text(7.5, 2, 'BUDGET: $20K → Low Value', ha='center', fontsize=12, fontweight='bold')
    
    ax4.set_title('Retention Strategy Action Plan', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/abstract_customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Abstract customer segmentation visualization created!")

def create_retention_strategy_flow():
    """Create abstract retention strategy flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'CUSTOMER RETENTION STRATEGY FLOW', ha='center', fontsize=18, fontweight='bold')
    
    # Customer Input
    customer_circle = Circle((2, 7), 0.8, color='lightblue', alpha=0.8)
    ax.add_patch(customer_circle)
    ax.text(2, 7, 'CUSTOMER\nBASE', ha='center', va='center', fontweight='bold')
    
    # Segmentation Process
    ax.arrow(2.8, 7, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    segment_box = Rectangle((4, 6.2), 2, 1.6, color='orange', alpha=0.7)
    ax.add_patch(segment_box)
    ax.text(5, 7, 'SEGMENTATION\nANALYSIS', ha='center', va='center', fontweight='bold')
    
    # Strategy Arrows
    ax.arrow(6, 7, 0.5, 1, head_width=0.2, head_length=0.2, fc='green', ec='green')
    ax.arrow(6, 7, 0.5, -1, head_width=0.2, head_length=0.2, fc='red', ec='red')
    ax.arrow(6, 7, 0.5, 0, head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    
    # VIP Strategy
    vip_box = Rectangle((7, 8.2), 2, 1.6, color='gold', alpha=0.8)
    ax.add_patch(vip_box)
    ax.text(8, 9, 'VIP\nENHANCEMENT', ha='center', va='center', fontweight='bold')
    
    # High Value High Risk Strategy
    hvhr_box = Rectangle((7, 5.2), 2, 1.6, color='red', alpha=0.7)
    ax.add_patch(hvhr_box)
    ax.text(8, 6, 'AGGRESSIVE\nRETENTION', ha='center', va='center', fontweight='bold', color='white')
    
    # Low Value Strategy
    lv_box = Rectangle((7, 2.2), 2, 1.6, color='lightgray', alpha=0.6)
    ax.add_patch(lv_box)
    ax.text(8, 3, 'NATURAL\nATTRITION', ha='center', va='center', fontweight='bold')
    
    # Results
    ax.arrow(9, 9, 1, 0, head_width=0.2, head_length=0.2, fc='green', ec='green')
    ax.arrow(9, 6, 1, 0, head_width=0.2, head_length=0.2, fc='green', ec='green')
    ax.arrow(9, 3, 1, 0, head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    
    results_box = Rectangle((10.2, 4.2), 1.6, 3.6, color='green', alpha=0.7)
    ax.add_patch(results_box)
    ax.text(11, 6, 'RESULTS\n• 242% ROI\n• $227K Revenue\n• 50% Churn\nReduction', ha='center', va='center', fontweight='bold', color='white')
    
    # Budget allocation
    ax.text(8, 1, 'BUDGET ALLOCATION: 80% → High Value, 20% → Low Value', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/retention_strategy_flow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Retention strategy flow diagram created!")

def create_roi_visualization():
    """Create abstract ROI visualization"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'RETENTION STRATEGY ROI BREAKDOWN', ha='center', fontsize=16, fontweight='bold')
    
    # Investment
    investment_circle = Circle((2, 7), 0.8, color='red', alpha=0.7)
    ax.add_patch(investment_circle)
    ax.text(2, 7, 'INVESTMENT\n$100K', ha='center', va='center', fontweight='bold', color='white')
    
    # ROI arrows
    ax.arrow(2.8, 7, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=3)
    
    # ROI breakdown
    # VIP ROI
    vip_roi = Circle((5, 8), 0.6, color='gold', alpha=0.8)
    ax.add_patch(vip_roi)
    ax.text(5, 8, 'VIP ROI\n440%', ha='center', va='center', fontweight='bold')
    
    # High Value High Risk ROI
    hvhr_roi = Circle((5, 6), 0.6, color='red', alpha=0.7)
    ax.add_patch(hvhr_roi)
    ax.text(5, 6, 'HVHR ROI\n190%', ha='center', va='center', fontweight='bold', color='white')
    
    # Low Value ROI
    lv_roi = Circle((5, 4), 0.6, color='lightgray', alpha=0.6)
    ax.add_patch(lv_roi)
    ax.text(5, 4, 'LV ROI\n75%', ha='center', va='center', fontweight='bold')
    
    # Total ROI
    ax.arrow(5.6, 6, 1, 0, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    
    total_roi = Circle((7.5, 6), 0.8, color='green', alpha=0.8)
    ax.add_patch(total_roi)
    ax.text(7.5, 6, 'TOTAL ROI\n242%', ha='center', va='center', fontweight='bold', color='white')
    
    # Revenue protection
    ax.text(7.5, 4, 'REVENUE PROTECTION\n$227K/month', ha='center', fontsize=12, fontweight='bold')
    
    # Strategy summary
    ax.text(2, 2, 'STRATEGY:\n• 80% budget to high value\n• Aggressive retention\n• Natural attrition', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/roi_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ ROI visualization created!")

def main():
    """Create all abstract visualizations"""
    print("🎨 Creating Abstract Visualizations for Customer Retention Strategy")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create visualizations
    create_abstract_customer_segmentation()
    create_retention_strategy_flow()
    create_roi_visualization()
    
    print("\n✅ All abstract visualizations created successfully!")
    print("📁 Files saved in: results/")
    print("   - abstract_customer_segmentation.png")
    print("   - retention_strategy_flow.png")
    print("   - roi_visualization.png")

if __name__ == "__main__":
    main() 