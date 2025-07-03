#!/usr/bin/env python3
"""
Summary Infographic: Complete Customer Churn Analysis Overview
Creates a comprehensive visual summary of all findings and strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from data_loader import TelcoDataLoader

def create_summary_infographic():
    """Create comprehensive summary infographic"""
    
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
    
    # Calculate key metrics
    segment_analysis = data.groupby('Segment').agg({
        'customerID': 'count',
        'MonthlyCharges': ['sum', 'mean'],
        'Churn': lambda x: (x == 'Yes').mean() * 100
    }).round(2)
    
    # Create infographic
    fig = plt.figure(figsize=(20, 16))
    
    # Main title
    fig.suptitle('TELCO CUSTOMER CHURN ANALYSIS: COMPLETE STRATEGY OVERVIEW', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Create grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Key Statistics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Key stats box
    stats_box = FancyBboxPatch((0.5, 6), 9, 3, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', alpha=0.8)
    ax1.add_patch(stats_box)
    
    ax1.text(5, 8.5, 'KEY STATISTICS', ha='center', fontsize=14, fontweight='bold')
    ax1.text(2, 7.5, f'Total Customers: {len(data):,}', ha='center', fontsize=12)
    ax1.text(8, 7.5, f'Churn Rate: 26.5%', ha='center', fontsize=12)
    ax1.text(2, 6.5, f'Monthly Revenue: ${data["MonthlyCharges"].sum():,.0f}', ha='center', fontsize=12)
    ax1.text(8, 6.5, f'Avg Monthly: ${data["MonthlyCharges"].mean():.0f}', ha='center', fontsize=12)
    
    # 2. Customer Segmentation (top right)
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Segmentation matrix
    ax2.text(5, 9, 'CUSTOMER SEGMENTATION MATRIX', ha='center', fontsize=14, fontweight='bold')
    
    # VIP segment
    vip_box = Rectangle((1, 6), 2, 2, color='gold', alpha=0.8)
    ax2.add_patch(vip_box)
    ax2.text(2, 7, 'VIP - KEEP\n1,472 customers\n$139K revenue\n10.7% churn', ha='center', va='center', fontweight='bold')
    
    # High Value High Risk
    hvhr_box = Rectangle((4, 6), 2, 2, color='red', alpha=0.7)
    ax2.add_patch(hvhr_box)
    ax2.text(5, 7, 'HIGH VALUE\nHIGH RISK\n2,043 customers\n$180K revenue\n52.8% churn', ha='center', va='center', fontweight='bold', color='white')
    
    # Low Value segments
    lvlr_box = Rectangle((1, 3), 2, 2, color='lightgray', alpha=0.6)
    ax2.add_patch(lvlr_box)
    ax2.text(2, 4, 'LOW VALUE\nLOW RISK\n1,696 customers\n$60K revenue\n3.3% churn', ha='center', va='center', fontweight='bold')
    
    lvhr_box = Rectangle((4, 3), 2, 2, color='darkgray', alpha=0.6)
    ax2.add_patch(lvhr_box)
    ax2.text(5, 4, 'LOW VALUE\nHIGH RISK\n1,832 customers\n$78K revenue\n31.4% churn', ha='center', va='center', fontweight='bold')
    
    # Labels
    ax2.text(5, 2, 'HIGH VALUE', ha='center', fontweight='bold')
    ax2.text(5, 1, 'LOW VALUE', ha='center', fontweight='bold')
    ax2.text(0.5, 5, 'LOW\nRISK', ha='center', va='center', fontweight='bold')
    ax2.text(9.5, 5, 'HIGH\nRISK', ha='center', va='center', fontweight='bold')
    
    # 3. Budget Allocation (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    ax3.text(5, 9, 'BUDGET ALLOCATION', ha='center', fontsize=14, fontweight='bold')
    
    # Budget pie chart
    budget_data = [50, 30, 20]
    budget_labels = ['High Value\nHigh Risk', 'VIP', 'Low Value']
    budget_colors = ['red', 'gold', 'lightgray']
    
    wedges, texts, autotexts = ax3.pie(budget_data, labels=budget_labels, colors=budget_colors, 
                                       autopct='%1.0f%%', startangle=90)
    ax3.text(5, 1, '$100K Total Budget', ha='center', fontweight='bold')
    
    # 4. Strategy Overview (middle left)
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    ax4.text(5, 9.5, 'RETENTION STRATEGY OVERVIEW', ha='center', fontsize=14, fontweight='bold')
    
    # Strategy boxes
    # VIP Strategy
    vip_strategy = Rectangle((0.5, 7), 4, 1.5, color='gold', alpha=0.8)
    ax4.add_patch(vip_strategy)
    ax4.text(2.5, 7.75, 'VIP STRATEGY', ha='center', fontweight='bold')
    ax4.text(2.5, 7.25, '• Enhanced benefits\n• Referral program\n• Brand advocacy', ha='center', fontsize=10)
    
    # High Value High Risk Strategy
    hvhr_strategy = Rectangle((5.5, 7), 4, 1.5, color='red', alpha=0.7)
    ax4.add_patch(hvhr_strategy)
    ax4.text(7.5, 7.75, 'AGGRESSIVE RETENTION', ha='center', fontweight='bold', color='white')
    ax4.text(7.5, 7.25, '• Emergency intervention\n• Contract conversion\n• Personalized offers', ha='center', fontsize=10, color='white')
    
    # Low Value Strategy
    lv_strategy = Rectangle((2.5, 5), 5, 1.5, color='lightgray', alpha=0.6)
    ax4.add_patch(lv_strategy)
    ax4.text(5, 5.75, 'NATURAL ATTRITION', ha='center', fontweight='bold')
    ax4.text(5, 5.25, '• Minimal effort\n• Standard service\n• Let go naturally', ha='center', fontsize=10)
    
    # 5. ROI Projections (middle right)
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    
    ax5.text(5, 9.5, 'ROI PROJECTIONS', ha='center', fontsize=14, fontweight='bold')
    
    # ROI bars
    segments = ['VIP', 'High Value\nHigh Risk', 'Low Value']
    roi_values = [440, 190, 75]
    colors = ['gold', 'red', 'lightgray']
    
    for i, (segment, roi, color) in enumerate(zip(segments, roi_values, colors)):
        height = roi / 50  # Scale for visualization
        bar = Rectangle((1 + i*2.5, 2), 1.5, height, color=color, alpha=0.8)
        ax5.add_patch(bar)
        ax5.text(1.75 + i*2.5, height + 2.5, f'{roi}%', ha='center', fontweight='bold')
        ax5.text(1.75 + i*2.5, 1.5, segment, ha='center', fontsize=10)
    
    ax5.text(5, 0.5, 'Total ROI: 242%', ha='center', fontweight='bold', fontsize=12)
    
    # 6. Key Insights (bottom left)
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    
    ax6.text(5, 9.5, 'KEY INSIGHTS', ha='center', fontsize=14, fontweight='bold')
    
    insights = [
        '🎯 80% of revenue comes from top 2 segments',
        '💔 Month-to-month contracts = 52.8% churn rate',
        '💰 High monthly charges correlate with churn',
        '⏰ New customers (<12 months) = higher risk',
        '📊 Contract type is the biggest churn predictor',
        '🎪 VIP customers have lowest churn (10.7%)'
    ]
    
    for i, insight in enumerate(insights):
        ax6.text(0.5, 8 - i*1.2, insight, fontsize=11)
    
    # 7. Action Plan (bottom right)
    ax7 = fig.add_subplot(gs[2, 2:4])
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    ax7.axis('off')
    
    ax7.text(5, 9.5, 'IMMEDIATE ACTION PLAN', ha='center', fontsize=14, fontweight='bold')
    
    actions = [
        '🚨 Week 1-2: Emergency intervention for high-risk customers',
        '💎 Month 1-2: Launch VIP enhancement program',
        '📋 Week 3-8: Contract conversion campaigns',
        '🌟 Month 3-6: Build long-term loyalty programs',
        '📊 Monthly: Monitor and optimize performance',
        '🎯 Target: 50% churn reduction in 6 months'
    ]
    
    for i, action in enumerate(actions):
        ax7.text(0.5, 8 - i*1.2, action, fontsize=11)
    
    # 8. Success Metrics (bottom)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.set_xlim(0, 10)
    ax8.set_ylim(0, 10)
    ax8.axis('off')
    
    ax8.text(5, 9, 'SUCCESS METRICS & EXPECTED OUTCOMES', ha='center', fontsize=14, fontweight='bold')
    
    # Metrics grid
    metrics = [
        ('Churn Reduction', '52.8% → 25%', '50% improvement'),
        ('Revenue Protection', '$227K/month', 'Protected revenue'),
        ('Total ROI', '242%', '$242K return on $100K'),
        ('Customer Satisfaction', '95%', 'VIP satisfaction target'),
        ('Contract Conversion', '30%', 'Month-to-month → Annual'),
        ('Cost Savings', '$15K/month', 'Reduced service costs')
    ]
    
    for i, (metric, value, description) in enumerate(metrics):
        x_pos = 1 + (i % 3) * 3
        y_pos = 7 - (i // 3) * 2
        
        metric_box = Rectangle((x_pos, y_pos), 2.5, 1.5, color='lightgreen', alpha=0.7)
        ax8.add_patch(metric_box)
        ax8.text(x_pos + 1.25, y_pos + 1, metric, ha='center', fontweight='bold', fontsize=10)
        ax8.text(x_pos + 1.25, y_pos + 0.5, value, ha='center', fontsize=12, color='darkgreen')
        ax8.text(x_pos + 1.25, y_pos + 0.1, description, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Summary infographic created!")

def main():
    """Create summary infographic"""
    print("📊 Creating Summary Infographic")
    print("=" * 40)
    
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    create_summary_infographic()
    
    print("\n✅ Summary infographic created successfully!")
    print("📁 File saved as: results/summary_infographic.png")

if __name__ == "__main__":
    main() 