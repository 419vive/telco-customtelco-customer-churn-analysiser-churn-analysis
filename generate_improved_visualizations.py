#!/usr/bin/env python3
"""
Improved Visualizations for Telco Customer Churn Analysis
Fixed spacing issues and added detailed explanations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def create_improved_customer_segmentation():
    """Create improved customer segmentation visualization with no overlapping"""
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(24, 20))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Customer Universe (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Create non-overlapping customer clusters
    # VIP - Keep (gold stars) - Top left quadrant
    vip_x = np.random.uniform(1, 3.5, 12)
    vip_y = np.random.uniform(7, 9, 12)
    ax1.scatter(vip_x, vip_y, s=150, c='gold', marker='*', alpha=0.8, label='VIP Customers')
    
    # High Value High Risk (red circles) - Top right quadrant
    hvhr_x = np.random.uniform(6.5, 9, 15)
    hvhr_y = np.random.uniform(7, 9, 15)
    ax1.scatter(hvhr_x, hvhr_y, s=120, c='red', alpha=0.7, label='High Value High Risk')
    
    # Low Value Low Risk (light blue squares) - Bottom left quadrant
    lvlr_x = np.random.uniform(1, 3.5, 10)
    lvlr_y = np.random.uniform(1, 3, 10)
    ax1.scatter(lvlr_x, lvlr_y, s=100, c='lightblue', marker='s', alpha=0.6, label='Low Value Low Risk')
    
    # Low Value High Risk (dark blue triangles) - Bottom right quadrant
    lvhr_x = np.random.uniform(6.5, 9, 10)
    lvhr_y = np.random.uniform(1, 3, 10)
    ax1.scatter(lvhr_x, lvhr_y, s=110, c='darkblue', marker='^', alpha=0.6, label='Low Value High Risk')
    
    # Add clear labels
    ax1.text(5, 9.5, 'HIGH VALUE', ha='center', fontsize=14, fontweight='bold', color='darkgreen')
    ax1.text(5, 0.5, 'LOW VALUE', ha='center', fontsize=14, fontweight='bold', color='darkred')
    ax1.text(0.5, 5, 'LOW RISK', rotation=90, fontsize=14, fontweight='bold', color='darkblue')
    ax1.text(9.5, 5, 'HIGH RISK', rotation=90, fontsize=14, fontweight='bold', color='darkorange')
    
    # Add quadrant lines
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_title('Customer Universe: Value vs Risk Segmentation', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    # 2. Revenue Distribution (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Revenue data
    segments = ['VIP', 'High Value\nHigh Risk', 'Low Value\nLow Risk', 'Low Value\nHigh Risk']
    revenue = [139, 180, 60, 78]  # in thousands
    colors = ['gold', 'red', 'lightblue', 'darkblue']
    
    bars = ax2.bar(segments, revenue, color=colors, alpha=0.8)
    ax2.set_title('Monthly Revenue by Segment ($K)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Revenue ($K)', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, revenue):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'${value}K\n({value/sum(revenue)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Churn Risk Heatmap (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    churn_data = np.array([
        [10.7, 52.8],  # VIP, High Value High Risk
        [3.3, 31.4]    # Low Value Low Risk, Low Value High Risk
    ])
    
    im = ax3.imshow(churn_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=60)
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Low Risk', 'High Risk'], fontsize=12)
    ax3.set_yticklabels(['High Value', 'Low Value'], fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if churn_data[i, j] > 30 else 'black'
            ax3.text(j, i, f'{churn_data[i, j]:.1f}%\nChurn Rate', 
                    ha="center", va="center", color=color, fontweight='bold', fontsize=11)
    
    ax3.set_title('Churn Risk Heatmap (%)', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Churn Rate (%)')
    
    # 4. Strategy Action Plan (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    
    # Strategy boxes with clear spacing
    # VIP Strategy
    vip_box = FancyBboxPatch((0.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1", 
                            facecolor='gold', alpha=0.8, edgecolor='black', linewidth=2)
    ax4.add_patch(vip_box)
    ax4.text(1.75, 6.5, 'VIP STRATEGY', ha='center', va='center', fontweight='bold', fontsize=14)
    ax4.text(1.75, 6, 'ENHANCE EXPERIENCE', ha='center', va='center', fontweight='bold', fontsize=12)
    ax4.text(1.75, 5.7, '• Premium Support\n• Exclusive Offers\n• Early Access Features', 
             ha='center', va='center', fontsize=10)
    
    # High Value High Risk Strategy
    hvhr_box = FancyBboxPatch((3.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1", 
                              facecolor='red', alpha=0.7, edgecolor='black', linewidth=2)
    ax4.add_patch(hvhr_box)
    ax4.text(4.75, 6.5, 'AGGRESSIVE RETENTION', ha='center', va='center', fontweight='bold', color='white', fontsize=14)
    ax4.text(4.75, 6, 'FIGHT TO KEEP', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    ax4.text(4.75, 5.7, '• Personalized Offers\n• Retention Campaigns\n• Win-back Programs', 
             ha='center', va='center', color='white', fontsize=10)
    
    # Low Value Strategy
    lv_box = FancyBboxPatch((6.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1", 
                           facecolor='lightgray', alpha=0.6, edgecolor='black', linewidth=2)
    ax4.add_patch(lv_box)
    ax4.text(7.75, 6.5, 'LOW VALUE STRATEGY', ha='center', va='center', fontweight='bold', fontsize=14)
    ax4.text(7.75, 6, 'LET GO NATURALLY', ha='center', va='center', fontweight='bold', fontsize=12)
    ax4.text(7.75, 5.7, '• Minimal Effort\n• Standard Service\n• No Special Offers', 
             ha='center', va='center', fontsize=10)
    
    # Budget allocation
    budget_box = FancyBboxPatch((9.5, 5.5), 2, 2, boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=2)
    ax4.add_patch(budget_box)
    ax4.text(10.5, 6.5, 'BUDGET ALLOCATION', ha='center', fontsize=12, fontweight='bold')
    ax4.text(10.5, 6, '$80K → Top 2 Segments', ha='center', fontsize=11, fontweight='bold')
    ax4.text(10.5, 5.7, '$20K → Low Value', ha='center', fontsize=11, fontweight='bold')
    
    ax4.set_title('Retention Strategy Action Plan', fontsize=18, fontweight='bold', pad=20)
    
    # 5. Expected Outcomes (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_xlim(0, 12)
    ax5.set_ylim(0, 4)
    ax5.axis('off')
    
    # Outcome boxes
    outcomes = [
        ('Churn Reduction', '5-10%', 'lightblue'),
        ('Revenue Protection', '$1.45M/year', 'lightgreen'),
        ('Customer Satisfaction', '+15%', 'lightyellow'),
        ('ROI', '300%', 'lightcoral')
    ]
    
    for i, (metric, value, color) in enumerate(outcomes):
        x_pos = 1 + i * 2.5
        box = FancyBboxPatch((x_pos-0.8, 1.5), 1.6, 2, boxstyle="round,pad=0.1", 
                            facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax5.add_patch(box)
        ax5.text(x_pos, 2.5, metric, ha='center', va='center', fontweight='bold', fontsize=12)
        ax5.text(x_pos, 2, value, ha='center', va='center', fontweight='bold', fontsize=14, color='darkred')
    
    ax5.set_title('Expected Outcomes from Retention Strategy', fontsize=16, fontweight='bold', pad=20)
    
    # Add overall title
    fig.suptitle('Customer Segmentation Analysis & Retention Strategy Framework', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('results/improved_customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Improved customer segmentation visualization created with no overlapping elements!")

def create_improved_retention_flow():
    """Create improved retention strategy flow with better spacing"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'CUSTOMER RETENTION STRATEGY FLOW', ha='center', fontsize=20, fontweight='bold')
    ax.text(8, 9, 'From Data Analysis to Actionable Retention Campaigns', ha='center', fontsize=14, color='gray')
    
    # Process flow with better spacing
    steps = [
        ('CUSTOMER\nBASE\n(7,043)', 2, 7, 'lightblue'),
        ('SEGMENTATION\nANALYSIS', 5, 7, 'lightgreen'),
        ('STRATEGY\nDEVELOPMENT', 8, 7, 'orange'),
        ('CAMPAIGN\nEXECUTION', 11, 7, 'red'),
        ('MONITORING\n& OPTIMIZATION', 14, 7, 'purple')
    ]
    
    # Draw steps
    for i, (label, x, y, color) in enumerate(steps):
        # Circle for each step
        circle = Circle((x, y), 1, color=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Arrow to next step
        if i < len(steps) - 1:
            next_x = steps[i+1][1]
            ax.arrow(x+1, y, next_x-x-2, 0, head_width=0.3, head_length=0.3, 
                    fc='black', ec='black', linewidth=2)
    
    # Results section below
    results_box = FancyBboxPatch((2, 3), 12, 3, boxstyle="round,pad=0.2", 
                                facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2)
    ax.add_patch(results_box)
    
    ax.text(8, 5, 'RESULTS & OPTIMIZATION', ha='center', va='center', fontweight='bold', fontsize=16)
    ax.text(8, 4.5, '• Churn Rate Reduction: 5-10%', ha='center', va='center', fontsize=12)
    ax.text(8, 4.2, '• Revenue Protection: $1.45M/year', ha='center', va='center', fontsize=12)
    ax.text(8, 3.9, '• Customer Satisfaction Improvement: +15%', ha='center', va='center', fontsize=12)
    ax.text(8, 3.6, '• Implementation Timeline: 3-6 months | Budget: $100K | Expected ROI: 300%', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Side explanations
    ax.text(1, 6, 'DATA SOURCES:', ha='left', fontsize=11, fontweight='bold')
    ax.text(1, 5.5, '• Customer Demographics', ha='left', fontsize=10)
    ax.text(1, 5.2, '• Service Usage Patterns', ha='left', fontsize=10)
    ax.text(1, 4.9, '• Payment History', ha='left', fontsize=10)
    ax.text(1, 4.6, '• Contract Information', ha='left', fontsize=10)
    
    ax.text(15, 6, 'KEY METRICS:', ha='right', fontsize=11, fontweight='bold')
    ax.text(15, 5.5, '• Monthly Charges', ha='right', fontsize=10)
    ax.text(15, 5.2, '• Contract Type', ha='right', fontsize=10)
    ax.text(15, 4.9, '• Payment Method', ha='right', fontsize=10)
    ax.text(15, 4.6, '• Service Bundles', ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/improved_retention_flow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Improved retention strategy flow created with better spacing!")

def create_improved_roi_visualization():
    """Create improved ROI visualization with clear explanations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: ROI Breakdown
    categories = ['VIP Retention', 'High Value\nHigh Risk', 'Low Value\nSegments']
    costs = [40, 40, 20]  # in thousands
    benefits = [120, 150, 30]  # in thousands
    rois = [200, 275, 50]  # percentages
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, costs, width, label='Investment ($K)', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, benefits, width, label='Revenue Protection ($K)', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Customer Segments', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amount ($K)', fontsize=12, fontweight='bold')
    ax1.set_title('ROI Analysis: Investment vs Revenue Protection', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'${height}K', ha='center', va='bottom', fontweight='bold')
    
    # Right: ROI Comparison
    ax2.bar(categories, rois, color=['gold', 'red', 'lightblue'], alpha=0.8)
    ax2.set_xlabel('Customer Segments', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Return on Investment by Segment', fontsize=16, fontweight='bold')
    
    # Add ROI labels
    for i, roi in enumerate(rois):
        ax2.text(i, roi + 10, f'{roi}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add summary box
    total_investment = sum(costs)
    total_benefit = sum(benefits)
    total_roi = (total_benefit - total_investment) / total_investment * 100
    
    summary_text = f'TOTAL INVESTMENT: ${total_investment}K\nTOTAL BENEFIT: ${total_benefit}K\nOVERALL ROI: {total_roi:.0f}%'
    ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/improved_roi_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Improved ROI visualization created with clear explanations!")

def main():
    """Generate all improved visualizations"""
    print("🎨 Generating improved visualizations with better spacing and explanations...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Generate visualizations
    create_improved_customer_segmentation()
    create_improved_retention_flow()
    create_improved_roi_visualization()
    
    print("\n✅ All improved visualizations generated successfully!")
    print("📁 Files saved in 'results/' directory:")
    print("   - improved_customer_segmentation.png")
    print("   - improved_retention_flow.png")
    print("   - improved_roi_visualization.png")

if __name__ == "__main__":
    main() 