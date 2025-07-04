#!/usr/bin/env python3
"""
Create a comprehensive infographic summarizing the First Principles Telco Churn Analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
import os

# Set up the plotting environment
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'

def create_comprehensive_infographic():
    """Create a comprehensive infographic of the analysis"""
    
    # Create a large figure
    fig = plt.figure(figsize=(20, 24))
    
    # Set up the layout
    gs = fig.add_gridspec(6, 4, height_ratios=[1, 1.5, 1.5, 1.5, 1, 1], hspace=0.3, wspace=0.2)
    
    # Title section
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.7, '🎯 TELCO CUSTOMER CHURN ANALYSIS', 
                  fontsize=32, fontweight='bold', ha='center', va='center',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    ax_title.text(0.5, 0.3, 'First Principles Approach: Understanding Why Customers Leave', 
                  fontsize=18, ha='center', va='center', style='italic')
    ax_title.axis('off')
    
    # Key metrics section
    ax_metrics = fig.add_subplot(gs[1, :])
    create_key_metrics_section(ax_metrics)
    
    # Core insights section
    ax_insights1 = fig.add_subplot(gs[2, 0])
    create_contract_insight(ax_insights1)
    
    ax_insights2 = fig.add_subplot(gs[2, 1])
    create_tenure_insight(ax_insights2)
    
    ax_insights3 = fig.add_subplot(gs[2, 2])
    create_model_performance(ax_insights3)
    
    ax_insights4 = fig.add_subplot(gs[2, 3])
    create_price_insight(ax_insights4)
    
    # Customer segmentation
    ax_segments = fig.add_subplot(gs[3, :])
    create_customer_segmentation(ax_segments)
    
    # Retention strategy
    ax_strategy = fig.add_subplot(gs[4, :])
    create_retention_strategy(ax_strategy)
    
    # Business impact
    ax_impact = fig.add_subplot(gs[5, :])
    create_business_impact(ax_impact)
    
    plt.tight_layout()
    
    # Save the infographic
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/telco_churn_infographic.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Comprehensive infographic saved to: results/telco_churn_infographic.png")

def create_key_metrics_section(ax):
    """Create the key metrics overview"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    
    # Background
    rect = FancyBboxPatch((0.2, 0.2), 9.6, 2.6, boxstyle="round,pad=0.1", 
                         facecolor='#f0f8ff', edgecolor='#4169e1', linewidth=2)
    ax.add_patch(rect)
    
    # Title
    ax.text(5, 2.5, 'KEY BUSINESS METRICS', fontsize=20, fontweight='bold', 
            ha='center', va='center')
    
    # Metrics boxes
    metrics = [
        ("7,043", "Total Customers", "#ff6b6b"),
        ("39.1%", "Churn Rate", "#ff6b6b"),
        ("$2.3M", "Annual Revenue Loss", "#ff6b6b"),
        ("76.3%", "Model Accuracy", "#4ecdc4"),
        ("0.825", "ROC-AUC Score", "#4ecdc4")
    ]
    
    x_positions = [1.5, 3, 4.5, 6.5, 8.5]
    
    for i, (value, label, color) in enumerate(metrics):
        x = x_positions[i]
        
        # Metric box
        rect = FancyBboxPatch((x-0.4, 0.8), 0.8, 1.2, boxstyle="round,pad=0.05", 
                             facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Value
        ax.text(x, 1.6, value, fontsize=16, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Label
        ax.text(x, 0.5, label, fontsize=10, fontweight='bold', 
                ha='center', va='center', wrap=True)
    
    ax.axis('off')

def create_contract_insight(ax):
    """Create contract type insight visualization"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9, 'CONTRACT TYPE EFFECT', fontsize=14, fontweight='bold', 
            ha='center', va='center')
    
    # Data
    contracts = ['Month-to-month', 'One year', 'Two year']
    churn_rates = [61.1, 16.1, 9.0]
    colors = ['#ff6b6b', '#ffd93d', '#4ecdc4']
    
    # Bar chart
    bars = ax.bar(range(len(contracts)), churn_rates, color=colors, alpha=0.8)
    
    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, churn_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 70)
    ax.set_xticks(range(len(contracts)))
    ax.set_xticklabels(contracts, rotation=45, ha='right')
    ax.set_ylabel('Churn Rate (%)')
    
    # Insight box
    ax.text(1.5, 45, '6.8x Higher\nChurn Risk', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

def create_tenure_insight(ax):
    """Create tenure insight visualization"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9, 'CUSTOMER LIFECYCLE', fontsize=14, fontweight='bold', 
            ha='center', va='center')
    
    # Data
    tenure_groups = ['New\n(0-12m)', 'Growing\n(13-24m)', 'Mature\n(25-36m)', 'Loyal\n(37m+)']
    churn_rates = [57.5, 42.5, 42.5, 28.0]
    colors = ['#ff6b6b', '#ff9f43', '#ffd93d', '#4ecdc4']
    
    # Line chart
    x_pos = range(len(tenure_groups))
    ax.plot(x_pos, churn_rates, marker='o', linewidth=3, markersize=8, color='#2c3e50')
    ax.fill_between(x_pos, churn_rates, alpha=0.3, color='#3498db')
    
    # Add percentage labels
    for i, rate in enumerate(churn_rates):
        ax.text(i, rate + 2, f'{rate}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 65)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tenure_groups)
    ax.set_ylabel('Churn Rate (%)')
    
    # Insight box
    ax.text(2.5, 45, '2.1x Higher\nChurn Risk', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8))

def create_model_performance(ax):
    """Create model performance visualization"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9, 'MODEL VALIDATION', fontsize=14, fontweight='bold', 
            ha='center', va='center')
    
    # Feature importance (top 5)
    features = ['Contract', 'Tenure', 'Total Charges', 'Monthly Charges', 'Payment Method']
    importance = [40.6, 16.5, 16.3, 12.4, 5.2]
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    
    # Horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color=colors, alpha=0.8)
    
    # Add percentage labels
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{imp}%', ha='left', va='center', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance (%)')
    ax.set_xlim(0, 45)
    
    # Validation check
    ax.text(20, 3.5, '✅ Confirms\nFirst Principles!', 
            fontsize=10, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

def create_price_insight(ax):
    """Create price sensitivity insight"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9, 'PRICE SENSITIVITY', fontsize=14, fontweight='bold', 
            ha='center', va='center')
    
    # Data
    price_groups = ['Low\n(<$35)', 'Medium\n($35-65)', 'High\n($65-95)', 'Premium\n($95+)']
    churn_rates = [37.4, 39.2, 39.7, 39.3]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    # Bar chart
    bars = ax.bar(range(len(price_groups)), churn_rates, color=colors, alpha=0.8)
    
    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, churn_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 45)
    ax.set_xticks(range(len(price_groups)))
    ax.set_xticklabels(price_groups)
    ax.set_ylabel('Churn Rate (%)')
    
    # Insight box
    ax.text(2, 30, 'Moderate\nPrice Effect', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))

def create_customer_segmentation(ax):
    """Create customer segmentation visualization"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Title
    ax.text(5, 5.5, 'CUSTOMER SEGMENTATION STRATEGY', fontsize=18, fontweight='bold', 
            ha='center', va='center')
    
    # Segments data
    segments = [
        ("🚨 High Value High Risk", "1,292 customers", "$117,951/month", "82.4% churn", "#ff6b6b"),
        ("🛡️ VIP - Keep Safe", "2,230 customers", "$202,877/month", "14.7% churn", "#4ecdc4"),
        ("🔄 Low Value Low Risk", "2,340 customers", "$110,118/month", "15.8% churn", "#6bcf7f"),
        ("💰 Low Value High Risk", "1,181 customers", "$58,879/month", "84.1% churn", "#ffd93d")
    ]
    
    # Create segment boxes
    x_positions = [1.2, 3.6, 6, 8.4]
    
    for i, (name, customers, revenue, churn, color) in enumerate(segments):
        x = x_positions[i]
        
        # Segment box
        rect = FancyBboxPatch((x-0.6, 1), 1.2, 3.5, boxstyle="round,pad=0.1", 
                             facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Segment info
        ax.text(x, 4, name, fontsize=9, fontweight='bold', 
                ha='center', va='center', wrap=True, color='white')
        ax.text(x, 3.3, customers, fontsize=8, fontweight='bold', 
                ha='center', va='center', color='white')
        ax.text(x, 2.8, revenue, fontsize=8, fontweight='bold', 
                ha='center', va='center', color='white')
        ax.text(x, 2.3, churn, fontsize=8, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Budget allocation
        budget_allocations = ["60%", "25%", "10%", "5%"]
        ax.text(x, 1.5, f"Budget: {budget_allocations[i]}", fontsize=10, fontweight='bold', 
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    ax.axis('off')

def create_retention_strategy(ax):
    """Create retention strategy overview"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    
    # Background
    rect = FancyBboxPatch((0.2, 0.2), 9.6, 3.6, boxstyle="round,pad=0.1", 
                         facecolor='#f8f9fa', edgecolor='#6c757d', linewidth=2)
    ax.add_patch(rect)
    
    # Title
    ax.text(5, 3.5, 'RETENTION STRATEGY FRAMEWORK', fontsize=16, fontweight='bold', 
            ha='center', va='center')
    
    # Strategy elements
    strategies = [
        ("🚨 URGENT ACTION", "Personal calls, 20-30% discounts,\nFree upgrades, Contract incentives"),
        ("🛡️ VIP PROTECTION", "Rewards program, Priority service,\nExclusive offers, Satisfaction surveys"),
        ("🔄 MAINTENANCE", "Automated campaigns, Bundle offers,\nSelf-service improvements"),
        ("💰 LET GO", "Natural attrition, Exit surveys,\nWin-back campaigns")
    ]
    
    x_positions = [1.5, 3.5, 6.5, 8.5]
    
    for i, (title, actions) in enumerate(strategies):
        x = x_positions[i]
        
        ax.text(x, 2.8, title, fontsize=10, fontweight='bold', 
                ha='center', va='center')
        ax.text(x, 1.5, actions, fontsize=8, 
                ha='center', va='center', wrap=True)
    
    ax.axis('off')

def create_business_impact(ax):
    """Create business impact summary"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    
    # Title
    ax.text(5, 3.5, 'EXPECTED BUSINESS IMPACT', fontsize=16, fontweight='bold', 
            ha='center', va='center')
    
    # Impact metrics
    impacts = [
        ("$707K", "Annual Revenue\nProtection", "#2ecc71"),
        ("300%+", "ROI on Retention\nInvestments", "#3498db"),
        ("50%", "Retention Rate\nImprovement", "#e74c3c"),
        ("18%", "High-Value Customers\nTargeted", "#f39c12")
    ]
    
    x_positions = [1.5, 3.5, 6.5, 8.5]
    
    for i, (value, label, color) in enumerate(impacts):
        x = x_positions[i]
        
        # Impact circle
        circle = plt.Circle((x, 2), 0.5, facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Value
        ax.text(x, 2.2, value, fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Label
        ax.text(x, 0.8, label, fontsize=9, fontweight='bold', 
                ha='center', va='center', wrap=True)
    
    ax.axis('off')

if __name__ == "__main__":
    print("🎨 Creating comprehensive telco churn analysis infographic...")
    create_comprehensive_infographic()
    print("🎉 Infographic creation completed!")