# Telco Customer Churn Analysis & Prediction

## 📋 專案企劃書
**詳細的專案企劃書請參考**: [PROJECT_PLAN.md](./PROJECT_PLAN.md)

## Project Overview
This project focuses on analyzing customer churn in the telecommunications industry and developing predictive models to identify customers at risk of leaving. The goal is to help businesses implement focused customer retention programs.

## Dataset Description
The dataset contains customer information including:
- **Churn Status**: Whether the customer left within the last month
- **Services**: Phone, multiple lines, internet, online security, online backup, device protection, tech support, streaming TV and movies
- **Account Information**: Tenure, contract type, payment method, paperless billing, monthly charges, total charges
- **Demographics**: Gender, age range, partner status, dependents

## Project Structure
```
project1/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── feature_engineering.py # Feature engineering utilities
│   ├── models.py             # ML model implementations
│   └── evaluation.py         # Model evaluation metrics
├── models/                   # Trained model files
├── results/                  # Analysis results and visualizations
└── config/                   # Configuration files
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
The project expects the IBM Telco Customer Churn dataset. You can download it from:
- [IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- Or use the provided data loading script

### 3. Run Analysis
```bash
# Run data exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Run complete analysis pipeline
python src/main.py
```

## Key Features
- **Comprehensive Data Analysis**: Exploratory data analysis with visualizations
- **Feature Engineering**: Advanced feature creation and selection
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, Neural Networks
- **Model Evaluation**: ROC curves, confusion matrices, feature importance
- **Business Insights**: Actionable recommendations for customer retention

## Business Impact
This analysis helps telecom companies:
- Identify customers at high risk of churning
- Understand key factors driving customer churn
- Develop targeted retention strategies
- Optimize customer service and marketing efforts

## 📊 專案進度追蹤

### ✅ 已完成階段
- **Phase 1: 商業理解** - 完成業務需求分析和目標定義
- **Phase 2: 數據理解** - 完成數據探索和質量評估
- **Phase 3: 數據準備** - 進行中，數據清洗和特徵工程

### 🔄 進行中階段
- 數據預處理和特徵工程
- 模型開發準備

### 📋 待完成階段
- **Phase 4: 建模** - 模型訓練和驗證
- **Phase 5: 評估** - 模型性能評估
- **Phase 6: 部署** - 模型部署和監控

## 🎯 關鍵指標
- **目標準確率**: > 80%
- **目標 ROC-AUC**: > 0.85
- **預期流失率降低**: 5-10%
- **預期收入保護**: $1,452,475/年

## Technologies Used
- Python 3.8+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Jupyter Notebooks
- XGBoost, LightGBM
- TensorFlow/Keras (for neural networks)

## License
This project is for educational and research purposes. 