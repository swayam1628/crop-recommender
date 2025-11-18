# ---------------------------------------------------------
# 🌾 CROP RECOMMENDATION SYSTEM – Premium UI (Streamlit)
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io

# ----------------------  PAGE CONFIG  ----------------------
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------  LOAD ARTIFACTS  ----------------------
@st.cache_resource
def load_resources():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_resources()

# ----------------------  CSS FOR UI  ----------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #2E7D32;
        text-align: center;
    }
    .sub-title {
        font-size: 17px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ecf9f1;
        border-left: 8px solid #2E7D32;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------  HEADER  ----------------------
st.markdown("<h1 class='main-title'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the best crop based on soil & climate conditions</p>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------  USER INPUT FIELDS  ----------------------
st.header("🌱 Enter Environmental Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("🌱 Nitrogen (N)", min_value=0.0, value=80.0)
    P = st.number_input("🌿 Phosphorus (P)", min_value=0.0, value=40.0)
with col2:
    K = st.number_input("🌾 Potassium (K)", min_value=0.0, value=40.0)
    temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)
with col3:
    humidity = st.number_input("💧 Humidity (%)", value=70.0)
    ph = st.number_input("⚗️ Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("🌧️ Rainfall (mm)", value=200.0)

st.markdown("---")

# ----------------------  PREDICTION BUTTON  ----------------------
if st.button("🌾 Recommend Best Crop", use_container_width=True):

    # Prepare input sample
    sample = [[N, P, K, temperature, humidity, ph, rainfall]]
    sample_scaled = scaler.transform(sample)

    pred = model.predict(sample_scaled)
    crop = label_encoder.inverse_transform(pred)[0]

    # Display Prediction
    st.markdown(
        f"""
        <div class='prediction-box'>
            🌱 Recommended Crop: <span style='color:#1B5E20;'>{crop.upper()}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------  FEATURE CONTRIBUTION GRAPH  ----------------------
    st.subheader("📊 Feature Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    features = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
    values = [N, P, K, temperature, humidity, ph, rainfall]

    ax.bar(features, values, color="#2E7D32")
    ax.set_ylabel("Value")
    ax.set_title("Input Feature Values")

    st.pyplot(fig)

    # ----------------------  DOWNLOAD REPORT  ----------------------
    st.subheader("📄 Download Prediction Report")

    report = f"""
    Crop Recommendation Report
    ---------------------------
    Nitrogen (N): {N}
    Phosphorus (P): {P}
    Potassium (K): {K}
    Temperature: {temperature}
    Humidity: {humidity}
    Soil pH: {ph}
    Rainfall: {rainfall}

    ➤ Recommended Crop: {crop.upper()}
    """

    buffer = io.BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)

    st.download_button(
        label="📥 Download Report",
        data=buffer,
        file_name="crop_recommendation_report.txt",
        mime="text/plain"
    )
