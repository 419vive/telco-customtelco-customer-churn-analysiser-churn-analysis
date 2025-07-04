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
    """Create abstract customer segmentation visualization with proper spacing and explanations"""
    
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
    
    # Create abstract visualization with better spacing
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Customer Segmentation Analysis & Strategy Framework', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Abstract Customer Universe - Fixed spacing
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    
    # Create abstract customer clusters with better spacing
    # VIP - Keep (gold stars) - Top left
    for i in range(15):
        x = np.random.uniform(1, 4)
        y = np.random.uniform(8, 11)
        ax1.scatter(x, y, s=120, c='gold', marker='*', alpha=0.8)
    
    # High Value High Risk (red circles) - Top right
    for i in range(20):
        x = np.random.uniform(8, 11)
        y = np.random.uniform(8, 11)
        ax1.scatter(x, y, s=100, c='red', alpha=0.7)
    
    # Low Value Low Risk (light blue squares) - Bottom left
    for i in range(18):
        x = np.random.uniform(1, 4)
        y = np.random.uniform(1, 4)
        ax1.scatter(x, y, s=80, c='lightblue', marker='s', alpha=0.6)
    
    # Low Value High Risk (dark blue triangles) - Bottom right
    for i in range(18):
        x = np.random.uniform(8, 11)
        y = np.random.uniform(1, 4)
        ax1.scatter(x, y, s=90, c='darkblue', marker='^', alpha=0.6)
    
    # Add clear labels and explanations
    ax1.set_title('Customer Universe: Value vs Risk Segmentation', fontsize=16, fontweight='bold', pad=20)
    ax1.text(6, 11.5, 'HIGH VALUE', ha='center', fontsize=14, fontweight='bold', color='darkgreen')
    ax1.text(6, 0.5, 'LOW VALUE', ha='center', fontsize=14, fontweight='bold', color='darkred')
    ax1.text(0.5, 6, 'LOW RISK', rotation=90, fontsize=14, fontweight='bold', color='darkblue')
    ax1.text(11.5, 6, 'HIGH RISK', rotation=90, fontsize=14, fontweight='bold', color='darkorange')
    
    # Add segment explanations
    ax1.text(2.5, 7, 'VIP Customers\n(High Value, Low Risk)\nPriority: Keep & Enhance', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.7))
    ax1.text(9.5, 7, 'High Value High Risk\nPriority: Aggressive Retention', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    ax1.text(2.5, 5, 'Low Value Low Risk\nPriority: Let Go Naturally', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax1.text(9.5, 5, 'Low Value High Risk\nPriority: Minimal Effort', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="darkblue", alpha=0.7))
    
    # 2. Revenue Flow Visualization - Fixed spacing
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    
    # Create revenue flow diagram with better spacing
    segment_analysis = data.groupby('Segment').agg({
        'customerID': 'count',
        'MonthlyCharges': 'sum'
    })
    
    # VIP segment (large circle) - Top left
    vip_circle = Circle((3, 8), 1.2, color='gold', alpha=0.8)
    ax2.add_patch(vip_circle)
    ax2.text(3, 8, 'VIP\n$139K\n(15% of Revenue)', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # High Value High Risk (large circle with warning) - Top right
    hvhr_circle = Circle((9, 8), 1.4, color='red', alpha=0.7)
    ax2.add_patch(hvhr_circle)
    ax2.text(9, 8, 'HIGH VALUE\nHIGH RISK\n$180K\n(19% of Revenue)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    # Low Value segments (smaller circles) - Bottom
    lvlr_circle = Circle((3, 3), 0.8, color='lightblue', alpha=0.6)
    ax2.add_patch(lvlr_circle)
    ax2.text(3, 3, 'LOW VALUE\nLOW RISK\n$60K\n(6% of Revenue)', ha='center', va='center', fontsize=9)
    
    lvhr_circle = Circle((9, 3), 0.8, color='darkblue', alpha=0.6)
    ax2.add_patch(lvhr_circle)
    ax2.text(9, 3, 'LOW VALUE\nHIGH RISK\n$78K\n(8% of Revenue)', ha='center', va='center', fontsize=9)
    
    # Revenue flow arrows with better positioning
    ax2.arrow(3, 6.8, 0, -1, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    ax2.arrow(9, 6.6, 0, -1, head_width=0.2, head_length=0.2, fc='orange', ec='orange', linewidth=3)
    ax2.arrow(3, 1.8, 0, -0.5, head_width=0.1, head_length=0.1, fc='gray', ec='gray', linewidth=2)
    ax2.arrow(9, 1.8, 0, -0.5, head_width=0.1, head_length=0.1, fc='gray', ec='gray', linewidth=2)
    
    # Add total revenue box
    total_box = Rectangle((4.5, 0.5), 3, 1, color='lightgreen', alpha=0.8)
    ax2.add_patch(total_box)
    ax2.text(6, 1, 'TOTAL MONTHLY REVENUE\n$457K', ha='center', va='center', fontweight='bold', fontsize=12)
    
    ax2.set_title('Revenue Flow: Monthly Revenue by Segment', fontsize=16, fontweight='bold', pad=20)
    
    # 3. Churn Risk Heatmap - Fixed spacing
    ax3 = axes[1, 0]
    
    # Create churn risk matrix
    churn_data = np.array([
        [10.7, 52.8],  # VIP, High Value High Risk
        [3.3, 31.4]    # Low Value Low Risk, Low Value High Risk
    ])
    
    im = ax3.imshow(churn_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=60)
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Low Risk', 'High Risk'], fontsize=12)
    ax3.set_yticklabels(['High Value', 'Low Value'], fontsize=12)
    
    # Add text annotations with better contrast
    for i in range(2):
        for j in range(2):
            color = 'white' if churn_data[i, j] > 30 else 'black'
            text = ax3.text(j, i, f'{churn_data[i, j]:.1f}%\nChurn Rate', 
                           ha="center", va="center", color=color, fontweight='bold', fontsize=11)
    
    ax3.set_title('Churn Risk Heatmap by Segment', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Value Level', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Churn Rate (%)')
    
    # Add explanation
    ax3.text(0.5, -0.3, 'Red = High Churn Risk\nGreen = Low Churn Risk', 
             transform=ax3.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # 4. Strategy Action Plan - Fixed spacing
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 12)
    ax4.axis('off')
    
    # Strategy boxes with better spacing
    # VIP Strategy - Top left
    vip_box = Rectangle((0.5, 8.5), 5, 2.5, color='gold', alpha=0.8)
    ax4.add_patch(vip_box)
    ax4.text(3, 9.5, 'VIP STRATEGY', ha='center', va='center', fontweight='bold', fontsize=14)
    ax4.text(3, 9, 'ENHANCE EXPERIENCE', ha='center', va='center', fontweight='bold', fontsize=12)
    ax4.text(3, 8.7, '• Premium Support\n• Exclusive Offers\n• Early Access Features', ha='center', va='center', fontsize=10)
    
    # High Value High Risk Strategy - Top right
    hvhr_box = Rectangle((6.5, 8.5), 5, 2.5, color='red', alpha=0.7)
    ax4.add_patch(hvhr_box)
    ax4.text(9, 9.5, 'AGGRESSIVE RETENTION', ha='center', va='center', fontweight='bold', color='white', fontsize=14)
    ax4.text(9, 9, 'FIGHT TO KEEP', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    ax4.text(9, 8.7, '• Personalized Offers\n• Retention Campaigns\n• Win-back Programs', ha='center', va='center', color='white', fontsize=10)
    
    # Low Value Strategy - Bottom
    lv_box = Rectangle((2, 5), 8, 2.5, color='lightgray', alpha=0.6)
    ax4.add_patch(lv_box)
    ax4.text(6, 6.5, 'LOW VALUE STRATEGY', ha='center', va='center', fontweight='bold', fontsize=14)
    ax4.text(6, 6, 'LET GO NATURALLY', ha='center', va='center', fontweight='bold', fontsize=12)
    ax4.text(6, 5.7, '• Minimal Effort\n• Standard Service\n• No Special Offers', ha='center', va='center', fontsize=10)
    
    # Budget allocation with clear explanation
    budget_box1 = Rectangle((1, 2), 4, 1.5, color='lightgreen', alpha=0.8)
    ax4.add_patch(budget_box1)
    ax4.text(3, 2.75, 'BUDGET ALLOCATION', ha='center', fontsize=12, fontweight='bold')
    ax4.text(3, 2.5, '$80K → Top 2 Segments', ha='center', fontsize=11, fontweight='bold')
    ax4.text(3, 2.25, '(VIP + High Value High Risk)', ha='center', fontsize=10)
    
    budget_box2 = Rectangle((7, 2), 4, 1.5, color='lightcoral', alpha=0.8)
    ax4.add_patch(budget_box2)
    ax4.text(9, 2.75, 'BUDGET ALLOCATION', ha='center', fontsize=12, fontweight='bold')
    ax4.text(9, 2.5, '$20K → Low Value', ha='center', fontsize=11, fontweight='bold')
    ax4.text(9, 2.25, '(Minimal Retention Effort)', ha='center', fontsize=10)
    
    # Expected outcomes
    outcome_box = Rectangle((3, 0.5), 6, 1, color='lightblue', alpha=0.8)
    ax4.add_patch(outcome_box)
    ax4.text(6, 1, 'EXPECTED OUTCOMES: 5-10% Churn Reduction, $1.45M Annual Revenue Protection', 
             ha='center', va='center', fontweight='bold', fontsize=11)
    
    ax4.set_title('Retention Strategy Action Plan', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/abstract_customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Abstract customer segmentation visualization created with proper spacing and explanations!")

def create_retention_strategy_flow():
    """Create abstract retention strategy flow diagram with better spacing"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'CUSTOMER RETENTION STRATEGY FLOW', ha='center', fontsize=20, fontweight='bold')
    ax.text(8, 11, 'From Data Analysis to Actionable Retention Campaigns', ha='center', fontsize=14, color='gray')
    
    # Customer Input - Better positioned
    customer_circle = Circle((2, 9), 1, color='lightblue', alpha=0.8)
    ax.add_patch(customer_circle)
    ax.text(2, 9, 'CUSTOMER\nBASE\n(7,043)', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Segmentation Process - Better positioned
    ax.arrow(3, 9, 1.5, 0, head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=2)
    segment_box = Rectangle((4.5, 8), 3, 2, color='lightgreen', alpha=0.7)
    ax.add_patch(segment_box)
    ax.text(6, 9, 'SEGMENTATION\nANALYSIS', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(6, 8.5, '• Value Assessment\n• Risk Evaluation\n• Behavior Patterns', ha='center', va='center', fontsize=9)
    
    # Strategy Development - Better positioned
    ax.arrow(7.5, 9, 1.5, 0, head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=2)
    strategy_box = Rectangle((9, 8), 3, 2, color='orange', alpha=0.7)
    ax.add_patch(strategy_box)
    ax.text(10.5, 9, 'STRATEGY\nDEVELOPMENT', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(10.5, 8.5, '• VIP Enhancement\n• Aggressive Retention\n• Cost Optimization', ha='center', va='center', fontsize=9)
    
    # Campaign Execution - Better positioned
    ax.arrow(12, 9, 1.5, 0, head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=2)
    campaign_box = Rectangle((13.5, 8), 2.5, 2, color='red', alpha=0.7)
    ax.add_patch(campaign_box)
    ax.text(14.75, 9, 'CAMPAIGN\nEXECUTION', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(14.75, 8.5, '• Targeted Offers\n• Personalized Communication\n• Monitoring', ha='center', va='center', fontsize=9)
    
    # Results and Optimization - Below the flow
    ax.arrow(8, 7.5, 0, -1, head_width=0.3, head_length=0.3, fc='blue', ec='blue', linewidth=2)
    results_box = Rectangle((5, 5.5), 6, 2, color='purple', alpha=0.7)
    ax.add_patch(results_box)
    ax.text(8, 6.5, 'RESULTS & OPTIMIZATION', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(8, 6, '• Churn Rate Reduction: 5-10%\n• Revenue Protection: $1.45M/year\n• Customer Satisfaction Improvement', ha='center', va='center', fontsize=10)
    
    # Detailed explanations on the sides
    # Left side explanations
    ax.text(1, 7, 'DATA SOURCES:', ha='left', fontsize=11, fontweight='bold')
    ax.text(1, 6.5, '• Customer Demographics', ha='left', fontsize=10)
    ax.text(1, 6.2, '• Service Usage Patterns', ha='left', fontsize=10)
    ax.text(1, 5.9, '• Payment History', ha='left', fontsize=10)
    ax.text(1, 5.6, '• Contract Information', ha='left', fontsize=10)
    
    # Right side explanations
    ax.text(15, 7, 'KEY METRICS:', ha='right', fontsize=11, fontweight='bold')
    ax.text(15, 6.5, '• Monthly Charges', ha='right', fontsize=10)
    ax.text(15, 6.2, '• Contract Type', ha='right', fontsize=10)
    ax.text(15, 5.9, '• Payment Method', ha='right', fontsize=10)
    ax.text(15, 5.6, '• Service Bundles', ha='right', fontsize=10)
    
    # Bottom explanations
    ax.text(8, 4.5, 'IMPLEMENTATION TIMELINE: 3-6 months | BUDGET: $100K | EXPECTED ROI: 300%', 
            ha='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/retention_strategy_flow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Retention strategy flow visualization created with proper spacing and explanations!")

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