**📱 Mobile Price Prediction using Machine Learning

🔍 Overview
This project predicts the launch price of smartphones based solely on their technical specifications using supervised regression models. It also includes a classification sub-model to categorize smartphones into Budget, Mid-range, and High-end tiers. The final product is deployed as a Streamlit web app for public interaction.

⚠️ Note: This project is for educational and portfolio purposes only. It does not account for discounts, depreciation, resale value, or promotional pricing.

Note: This project is strictly for educational and portfolio purposes. It does not account for real-world factors such as discounts, depreciation, resale value, or promotional pricing.

🧾 Dataset Description
Source: Compiled from public smartphone specifications (e-commerce and spec websites).
Samples: 1,370 smartphones
Target Variable: 📉 Launch Price (in INR)
Features Include:

**🔢 Features Include:
📶 RAM, Internal/External Memory, Battery (mAh), Fast Charging (W)
📱 Display Size, Screen Width/Height, Notch Type
📸 Camera Specs – Rear MP, Front MP, Rear Camera Count
📡 Network Capabilities – 3G, 4G, 5G, SIM Type
🤖 Android Version, 📦 Brand, ⚙️ Processor Name & Tier**


🧹 Data Preprocessing & Feature Engineering
🔄 Parsed text fields like "6000 mAh Battery" into numeric columns.
🏷️ Processor Tier Mapping:
Entry, Mid, Upper-Mid, Flagship, Ultra

🧠 Feature Engineering:
SIM capability (3G, 4G, 5G)
Company Encoding
Notch Type from screen resolution patterns

🧼 Missing Value Treatment:
Imputed using median/mode
Removed if non-essential
🔢 Label Encoding for categorical features

🧠 Modeling Approach
🧪 Regression Models:
🥇 XGBoost (Best)
🐈 CatBoost
💡 LightGBM
🌲 Random Forest
📈 Gradient Boosting
⚠️ AdaBoost

🧮 Evaluation Metrics:
🔢 MAE (Mean Absolute Error)
📉 RMSE (Root Mean Squared Error)
📊 R² and Adjusted R²

🏆 Best Performing Model: XGBoost
Metric	Value
🎯 Test MAE	₹6,308
📉 Test RMSE	₹10,608
📈 Test R²	0.8752
✅ Adjusted R²	0.8658
🔍 SHAP & Feature Importance

🧠 SHAP & Feature Importance
🔍 SHAP values were used to explain feature-level impact.
🥇 Top contributors to price prediction:
🖥️ Display Size
🧠 RAM
🔋 Battery
⚙️ Processor Tier
🏢 Company
📏 Screen Resolution

🚀 Streamlit Deployment
🌐 Interactive user interface
📱 Input real smartphone specs
💰 Predict launch price and class (Budget / Mid / High-end)
📊 View SHAP & feature influence visuals

✅ Confidence scores with bar charts
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
Special thanks to communities like Kaggle, StackOverflow, and HuggingFace for educational resources.

**
