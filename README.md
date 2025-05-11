Mobile Price Prediction using Machine Learning

Overview

This project aims to predict the launch price of smartphones based solely on their technical specifications using supervised regression models. It also includes a classification sub-model to categorize smartphones into Budget, Mid-range, and High-end tiers for comparative analysis. The final output is deployed as a Streamlit app for public interaction.

Note: This project is strictly for educational and portfolio purposes. It does not account for real-world factors such as discounts, depreciation, resale value, or promotional pricing.

📊 Dataset Description

Source: Scraped and compiled from e-commerce and specification sites.

Total Samples: 1370 smartphones

Target Variable: Price (Launch price in INR)

Features Include:

RAM, Internal/External Memory, Battery (mAh), Fast Charging (W)

Display size, Screen resolution, Notch type

Camera specs (Rear and Front MP, Camera Count)

SIM features (3G/4G/5G support, Dual/Single/No SIM)

Android version, Brand, Processor name & tier

🔧 Data Preprocessing

Parsing textual specs (e.g., "6000 mAh Battery") into numerical columns.

Feature engineering:

Processor Tier Mapping (Entry, Mid, Upper-Mid, Flagship, Ultra)

Company Brand Encoding

Network capabilities (3G, 4G, 5G) from SIM info

Notch type from screen resolution descriptions

Missing value handling:

Imputed using mode, median or removed if non-essential

Label Encoding for categorical variables like Company, Processor Tier

📈 Modeling Approach

Regression Models Used:

CatBoost Regressor ✅ (Best performer overall)

LightGBM

XGBoost

Random Forest

Gradient Boosting

AdaBoost

Model Evaluation Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score and Adjusted R²

🏆 Best Model Performance (XGBoost):

Test MAE: ₹6308

Test RMSE: ₹10608

Test R²: 0.8752

Adjusted R²: 0.8658

🔍 SHAP & Feature Importance

SHAP values were used to explain model outputs.

Key features influencing price:

Display size

RAM

Screen Resolution

Battery Capacity

Company & Processor Tier

💻 Streamlit Deployment

The entire system is deployed using Streamlit, offering:

Interactive UI for smartphone specs

Real-time price and tier prediction

Model confidence and feature breakdowns

📦 Project Structure

mobile-price-prediction/
├── app.py                     # Streamlit app
├── cleaned_data_mobile.csv   # Final cleaned dataset
├── XGBoost.pkl               # Final regression model
├── Processor_name.pkl        # Pickled processor names
├── Processor_tiers.pkl       # Mapping for processor tiers
├── PhoneName.pkl             # Pickled phone names
├── SHAP.png                  # SHAP plot for CatBoost
├── feature_importance.png    # Top feature chart
├── Model Comparison.png      # Bar chart of Adjusted R²
├── Shap Analysis.png         # SHAP summary
└── README.md                 # This file

📈 Future Improvements

Add ordinal regression with labels (Budget < Mid < High) for better tier handling.

Introduce manual label corrections (e.g., G85 with HD+ plastic body = Entry).

Use real-world launch price ranges scraped dynamically.

Explore transformers or tabular deep learning models (TabNet, FT-Transformer).

🤝 Acknowledgements

Inspired by real-world mobile shopping use cases.

Special thanks to communities like Kaggle, StackOverflow, and HuggingFace for educational resources.

