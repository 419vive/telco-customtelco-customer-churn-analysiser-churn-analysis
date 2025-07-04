#!/usr/bin/env python3
"""
Create comprehensive final infographic for Telco Churn Analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

def create_final_infographic():
    """Create a comprehensive infographic summarizing all findings"""
    
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    ax.axis('off')
    
    # Color palette
    colors = {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'accent': '#e74c3c',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'info': '#9b59b6',
        'light': '#ecf0f1',
        'dark': '#34495e'
    }
    
    # Title
    title_box = FancyBboxPatch((5, 110), 90, 8,
                              boxstyle="round,pad=0.5",
                              facecolor=colors['primary'],
                              edgecolor='none',
                              transform=ax.transData)
    ax.add_patch(title_box)
    ax.text(50, 114, 'TELCO CUSTOMER CHURN ANALYSIS', 
           fontsize=32, fontweight='bold', ha='center', va='center', color='white')
    ax.text(50, 111, 'First Principles Approach to Understanding & Preventing Customer Churn', 
           fontsize=16, ha='center', va='center', color='white', style='italic')
    
    # Executive Summary Box
    exec_box = FancyBboxPatch((5, 95), 42, 12,
                             boxstyle="round,pad=0.3",
                             facecolor=colors['secondary'],
                             alpha=0.9,
                             edgecolor=colors['dark'],
                             linewidth=2)
    ax.add_patch(exec_box)
    
    ax.text(26, 103, '📊 EXECUTIVE SUMMARY', fontsize=18, fontweight='bold', 
           ha='center', color='white')
    ax.text(26, 100.5, '7,043 Customers Analyzed', fontsize=14, ha='center', color='white')
    ax.text(26, 98.5, '35.5% Current Churn Rate', fontsize=14, ha='center', color='white')
    ax.text(26, 96.5, '$2.56M Annual Revenue at Risk', fontsize=14, ha='center', color='white', fontweight='bold')
    
    # Model Performance Box
    model_box = FancyBboxPatch((53, 95), 42, 12,
                              boxstyle="round,pad=0.3",
                              facecolor=colors['success'],
                              alpha=0.9,
                              edgecolor=colors['dark'],
                              linewidth=2)
    ax.add_patch(model_box)
    
    ax.text(74, 103, '🤖 MODEL PERFORMANCE', fontsize=18, fontweight='bold', 
           ha='center', color='white')
    ax.text(74, 100.5, 'Best Model: Logistic Regression', fontsize=14, ha='center', color='white')
    ax.text(74, 98.5, 'ROC-AUC Score: 0.821', fontsize=14, ha='center', color='white')
    ax.text(74, 96.5, '74.6% Accuracy', fontsize=14, ha='center', color='white', fontweight='bold')
    
    # First Principles Findings
    y_pos = 88
    ax.text(50, y_pos, '🔍 FIRST PRINCIPLES FINDINGS', fontsize=24, fontweight='bold', 
           ha='center', color=colors['primary'])
    
    # Finding 1: Contract Flexibility
    finding1_box = FancyBboxPatch((5, 70), 28, 15,
                                 boxstyle="round,pad=0.3",
                                 facecolor=colors['accent'],
                                 alpha=0.9)
    ax.add_patch(finding1_box)
    
    ax.text(19, 82, '📅 CONTRACT FLEXIBILITY', fontsize=16, fontweight='bold', 
           ha='center', color='white')
    ax.text(19, 79, 'PARADOX', fontsize=14, fontweight='bold', 
           ha='center', color='white')
    
    # Contract stats
    ax.text(19, 76, 'Month-to-month: 52.8% churn', fontsize=12, ha='center', color='white')
    ax.text(19, 74, 'Two-year: 4.8% churn', fontsize=12, ha='center', color='white')
    ax.text(19, 71.5, '↑ 47.9 percentage points', fontsize=12, ha='center', 
           color='white', fontweight='bold')
    
    # Finding 2: Customer Lifecycle
    finding2_box = FancyBboxPatch((36, 70), 28, 15,
                                 boxstyle="round,pad=0.3",
                                 facecolor=colors['warning'],
                                 alpha=0.9)
    ax.add_patch(finding2_box)
    
    ax.text(50, 82, '⏱️ CUSTOMER LIFECYCLE', fontsize=16, fontweight='bold', 
           ha='center', color='white')
    ax.text(50, 79, 'PATTERN', fontsize=14, fontweight='bold', 
           ha='center', color='white')
    
    ax.text(50, 76, 'Churned avg: 23.5 months', fontsize=12, ha='center', color='white')
    ax.text(50, 74, 'Retained avg: 31.0 months', fontsize=12, ha='center', color='white')
    ax.text(50, 71.5, 'Critical: First 24 months', fontsize=12, ha='center', 
           color='white', fontweight='bold')
    
    # Finding 3: Value Perception
    finding3_box = FancyBboxPatch((67, 70), 28, 15,
                                 boxstyle="round,pad=0.3",
                                 facecolor=colors['info'],
                                 alpha=0.9)
    ax.add_patch(finding3_box)
    
    ax.text(81, 82, '💰 VALUE PERCEPTION', fontsize=16, fontweight='bold', 
           ha='center', color='white')
    ax.text(81, 79, 'GAP', fontsize=14, fontweight='bold', 
           ha='center', color='white')
    
    ax.text(81, 76, 'Churners pay $9.27 more/month', fontsize=12, ha='center', color='white')
    ax.text(81, 74, 'Higher price ≠ Higher value', fontsize=12, ha='center', color='white')
    ax.text(81, 71.5, 'Need value enhancement', fontsize=12, ha='center', 
           color='white', fontweight='bold')
    
    # Customer Segmentation
    y_pos = 62
    ax.text(50, y_pos, '📊 CUSTOMER RISK SEGMENTATION', fontsize=22, fontweight='bold', 
           ha='center', color=colors['primary'])
    
    # Risk segments visualization
    segments = [
        ('Very Low', 38.0, 8.2, colors['success']),
        ('Low', 20.6, 31.4, colors['secondary']),
        ('Medium', 19.5, 53.5, colors['warning']),
        ('High', 17.7, 67.2, colors['accent']),
        ('Very High', 4.2, 84.7, colors['dark'])
    ]
    
    x_start = 10
    for i, (risk, pct, churn, color) in enumerate(segments):
        width = pct * 0.8  # Scale to fit
        
        # Segment bar
        seg_box = Rectangle((x_start, 52), width, 6,
                          facecolor=color,
                          alpha=0.8,
                          edgecolor=colors['dark'],
                          linewidth=2)
        ax.add_patch(seg_box)
        
        # Labels
        ax.text(x_start + width/2, 55, f'{risk}\nRisk', 
               fontsize=10, ha='center', va='center', color='white', fontweight='bold')
        ax.text(x_start + width/2, 50, f'{pct}%', 
               fontsize=9, ha='center', va='top')
        ax.text(x_start + width/2, 48.5, f'{churn}% churn', 
               fontsize=8, ha='center', va='top')
        
        x_start += width + 2
    
    # Business Impact
    y_pos = 42
    ax.text(50, y_pos, '💰 BUSINESS IMPACT SCENARIOS', fontsize=22, fontweight='bold', 
           ha='center', color=colors['primary'])
    
    # Impact scenarios
    scenarios = [
        ('Conservative\n5% Reduction', '$127,905', '2x ROI', colors['secondary']),
        ('Moderate\n7.5% Reduction', '$191,858', '3x ROI', colors['success']),
        ('Aggressive\n10% Reduction', '$255,811', '4x ROI', colors['accent'])
    ]
    
    for i, (scenario, revenue, roi, color) in enumerate(scenarios):
        x = 20 + i * 30
        
        # Scenario circle
        circle = Circle((x, 32), 8, facecolor=color, alpha=0.8,
                       edgecolor=colors['dark'], linewidth=3)
        ax.add_patch(circle)
        
        ax.text(x, 34, scenario, fontsize=11, ha='center', va='center', 
               color='white', fontweight='bold')
        ax.text(x, 30.5, revenue, fontsize=12, ha='center', va='center', 
               color='white', fontweight='bold')
        ax.text(x, 28.5, roi, fontsize=10, ha='center', va='center', 
               color='white')
        ax.text(x, 23, 'Annual Revenue\nProtected', fontsize=9, ha='center', va='center')
    
    # Top Churn Drivers
    y_pos = 18
    ax.text(25, y_pos, '🎯 TOP CHURN DRIVERS', fontsize=18, fontweight='bold', 
           ha='center', color=colors['primary'])
    
    drivers = [
        ('Contract Duration', 1.184),
        ('Tenure', 0.589),
        ('Internet Service', 0.354),
        ('Electronic Payment', 0.347),
        ('Monthly Charges', 0.273)
    ]
    
    y_start = 14
    for driver, importance in drivers:
        # Importance bar
        bar_width = importance * 20
        bar = Rectangle((10, y_start), bar_width, 1.5,
                       facecolor=colors['secondary'],
                       alpha=0.8)
        ax.add_patch(bar)
        
        ax.text(9, y_start + 0.75, driver, fontsize=10, ha='right', va='center')
        ax.text(10 + bar_width + 0.5, y_start + 0.75, f'{importance:.3f}', 
               fontsize=10, ha='left', va='center', fontweight='bold')
        
        y_start -= 2.5
    
    # Retention Strategies
    y_pos = 18
    ax.text(75, y_pos, '🚀 RETENTION STRATEGIES', fontsize=18, fontweight='bold', 
           ha='center', color=colors['primary'])
    
    strategies = [
        ('High-Risk Segments', '20% discount for contract upgrade', '15-20%', colors['accent']),
        ('Value Enhancement', 'Bundle services at 30% discount', '10%', colors['warning']),
        ('Early Intervention', 'Proactive milestone touchpoints', '25%', colors['success']),
        ('Payment Optimization', '$5/month auto-pay discount', '5%', colors['info'])
    ]
    
    y_start = 14
    for strategy, action, impact, color in strategies:
        # Strategy box
        strat_box = FancyBboxPatch((55, y_start - 2), 38, 2,
                                  boxstyle="round,pad=0.1",
                                  facecolor=color,
                                  alpha=0.3,
                                  edgecolor=color,
                                  linewidth=2)
        ax.add_patch(strat_box)
        
        ax.text(56, y_start - 0.5, strategy, fontsize=10, fontweight='bold', va='center')
        ax.text(56, y_start - 1.5, action, fontsize=8, va='center')
        ax.text(92, y_start - 1, f'{impact} churn ↓', fontsize=9, ha='right', 
               va='center', fontweight='bold', color=color)
        
        y_start -= 3.5
    
    # Call to Action
    cta_box = FancyBboxPatch((10, 0.5), 80, 5,
                            boxstyle="round,pad=0.3",
                            facecolor=colors['primary'],
                            edgecolor='none')
    ax.add_patch(cta_box)
    
    ax.text(50, 3.5, '🎯 IMPLEMENTATION ROADMAP', fontsize=16, fontweight='bold', 
           ha='center', color='white')
    ax.text(50, 1.5, '1. Deploy predictive model  →  2. Segment customers  →  3. Launch targeted campaigns  →  4. Monitor & optimize', 
           fontsize=12, ha='center', color='white')
    
    # Save the infographic
    plt.tight_layout()
    plt.savefig('results/comprehensive_churn_analysis_infographic.png', 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Comprehensive infographic created!")

if __name__ == "__main__":
    create_final_infographic()