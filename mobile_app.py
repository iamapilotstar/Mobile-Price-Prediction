import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Page config
st.set_page_config(page_title="📱 Mobile Price Prediction", layout="wide")

st.title("📱 Mobile Price Prediction")

st.markdown("""
<div style="font-size:18px; line-height:1.6;">
    <strong>⚠️ Disclaimer:</strong><br>
    Disclaimer:
Mobile pricing is influenced by brand, build quality, marketing, and regional strategies 
— not just hardware specs. 
This app estimates launch-tier positioning based on internal specifications.
The tool is intended strictly for educational and demonstration purposes. 
showcasing the end-to-end lifecycle of a machine learning project.  
Please do not use for real-world pricing decisions.
</div>
""", unsafe_allow_html=True)

# Sidebar - Model evaluation summary
st.sidebar.title("📊 Model Evaluation Summary")
model_scores = {
    "CatBoost": {"adj_r2": 0.8493, "mae_gap": 3492.10},
    "LightGBM": {"adj_r2": 0.8281, "mae_gap": 2338.29},
    "XGBoost": {"adj_r2": 0.8559, "mae_gap": 4543.00},
    "Random Forest": {"adj_r2": 0.8449, "mae_gap": 2704.40},
    "Gradient Boosting": {"adj_r2": 0.8242, "mae_gap": 3862.34}
}


for model, data in model_scores.items():
    st.sidebar.markdown(f"**{model}**")
    st.sidebar.write(f"Adjusted R²: `{data['adj_r2']:.4f}`")
    st.sidebar.write(f"MAE Gap (Test - Train): `{data['mae_gap']:.2f}`")

# Load models
@st.cache_data
def load_models():
    model_files = {
        "CatBoost": "CatBoost.pkl",
        "Gradient Boosting": "Gradient Boosting.pkl",
        "XGBoost": "XGBoost.pkl",
        "LightGBM": "LightGBM.pkl",
        "Random Forest": "Random Forest.pkl"
    }
    models = {}
    for name, file in model_files.items():
        try:
            with open(file, "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file}: {e}")
    return models

models = load_models()
if not models:
    st.error("❌ No models loaded. Please check your .pkl files.")
    st.stop()

# Load auxiliary data
with open("Processor_name.pkl", "rb") as f:
    processor_names = pickle.load(f)
with open("PhoneName.pkl", "rb") as f:
    phone_names = pickle.load(f)
with open("Processor_tiers.pkl", "rb") as f:
    processor_tiers = pickle.load(f)

# Mappings
brand_list = ['Samsung', 'Realme', 'Xiaomi', 'iQOO', 'Motorola', 'Vivo', 'OPPO', 'Google', 'Nothing', 'OnePlus']
sim_map = {'Single': 0, 'Dual': 1, 'No Sim': 2, 'Unknown': 3}
tier_label = {'Entry': 0, 'Mid': 1, 'Upper-Mid': 2, 'Flagship': 3, 'Ultra': 4, 'Unknown': -1}

# User Inputs
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.subheader("📱 Display & Build")
    display = st.text_input("Display Size (inches)", value="6.5")
    screen_width = st.text_input("Screen Width (px)", value="1080")
    screen_height = st.text_input("Screen Height (px)", value="2400")
    battery = st.text_input("Battery Capacity (mAh)", value="4500")
    fast_charging = st.text_input("Fast Charging Wattage (W)", value="30")
    sim_3g = st.selectbox("Supports 3G", [0, 1])
    sim_4g = st.selectbox("Supports 4G", [0, 1])
    sim_5g = st.selectbox("Supports 5G", [0, 1])

with col2:
    st.subheader("📷 Camera Specs")
    rear_camera = st.text_input("Rear Camera MP", value="50")
    rear_count = st.selectbox("Rear Camera Count", [1, 2, 3, 4])
    front_camera = st.text_input("Front Camera MP", value="16")
    brand = st.selectbox("Brand", sorted(brand_list))
    phone_model = st.selectbox("Phone Model", sorted(phone_names))

with col3:
    st.subheader("⚙️ Performance & OS")
    ram = st.selectbox("RAM (GB)", [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 18, 24])
    internal_memory = st.selectbox("Internal Memory (GB)", [16, 32, 64, 128, 256, 512])
    external_memory = st.selectbox("External Memory (GB)", [0, 64, 128, 256])
    android_version = st.selectbox("Android Version", [9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    sim_type = st.selectbox("SIM Type", list(sim_map.keys()))
    processor = st.selectbox("Processor Name", sorted(processor_names))

# Tier inference
processor_tier = processor_tiers.get(processor, 'Unknown')
processor_tier_encoded = tier_label.get(processor_tier, -1)

# Input construction
test_input = {
    'Sim Type': sim_map[sim_type],
    'Sim_3G': int(sim_3g),
    'Sim_4G': int(sim_4g),
    'Sim_5G': int(sim_5g),
    'RAM (GB)': ram,
    'Battery (mAh)': float(battery),
    'Display (inches)': float(display),
    'Primary Camera (MP)': float(rear_camera),
    'Primary_Camera_count': rear_count,
    'Front Camera (MP)': float(front_camera),
    'External Memory (GB)': external_memory,
    'Android Version': android_version,
    'Fast Charging (W)': float(fast_charging),
    'Internal Memory (GB)': internal_memory,
    'Screen Width': float(screen_width),
    'Screen Height': float(screen_height),
    'Notch Type': 0,
    'Company': brand_list.index(brand),
    'Processor Tier': processor_tier_encoded
}
input_df = pd.DataFrame([test_input])
st.markdown("---")

# Model Selection
selected_model = st.selectbox("Select a Model", list(models.keys()) + ["Ensemble (Average)"])
if st.button("💰 Predict Price"):
    try:
        if selected_model == "Ensemble (Average)":
            predictions = [model.predict(input_df)[0] for model in models.values()]
            prediction = round(np.mean(predictions) / 500) * 500
            st.success(f"📱 Ensemble Predicted Price: ₹{int(prediction):,}")
        else:
            prediction = models[selected_model].predict(input_df)[0]
            prediction = round(prediction / 500) * 500
            st.success(f"📱 Predicted Mobile Price: ₹{int(prediction):,}")
        st.caption(f"Processor Tier: **{processor_tier}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Feature Importance (if available)
st.subheader("📊 Top Features Affecting Price")
if os.path.exists("feature_importance.png"):
    st.image("feature_importance.png", caption="Feature Importance", width=700)
else:
    st.info("Upload 'feature_importance.png' to show importance plot.")
