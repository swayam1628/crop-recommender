# ---------------------------------------------------------
# 🌾 CROP RECOMMENDATION SYSTEM – Premium UI (Streamlit)
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import requests
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
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ecf9f1;
        border-left: 8px solid #2E7D32;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------  HEADER  ----------------------
st.markdown("<h1 class='main-title'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------  USER INPUT FIELDS  ----------------------
st.header("🌱 Enter Environmental Parameters")

# Line 1 → N, P, K
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("🌱 Nitrogen (N)", min_value=0.0, value=80.0)
with col2:
    P = st.number_input("🌿 Phosphorus (P)", min_value=0.0, value=40.0)
with col3:
    K = st.number_input("🌾 Potassium (K)", min_value=0.0, value=40.0)

# Line 2 → Humidity, Rainfall, Soil pH
col4, col5, col6 = st.columns(3)
with col4:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, value=70.0)
with col5:
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, value=200.0)
with col6:
    ph = st.number_input("⚗️ Soil pH", min_value=0.0, max_value=14.0, value=6.5)

# Line 3 → Temperature Centered
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)

st.markdown("---")

# ----------------------  PREDICTION BUTTON & SAFE OUTPUTS  ----------------------
if st.button("🌾 Recommend Best Crop", use_container_width=True):

    # Create input
    sample = [[N, P, K, temperature, humidity, ph, rainfall]]
    sample_scaled = scaler.transform(sample)

    # Prediction
    pred = model.predict(sample_scaled)
    crop = label_encoder.inverse_transform(pred)[0]

    # Display main crop
    st.markdown(
        f"""
        <div class='prediction-box'>
            🌱 <strong>Recommended Crop:</strong> 
            <span style='color:#1B5E20;'>{crop.upper()}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------- TOP 3 RECOMMENDATIONS (safe) ----------------------
    # initialize defaults so variables always exist
    top3_labels = ["N/A", "N/A", "N/A"]
    top3_scores = [0.0, 0.0, 0.0]

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(sample_scaled)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            top3_labels = label_encoder.inverse_transform(top3_idx)
            top3_scores = proba[top3_idx]
        except Exception as e:
            st.warning(f"Could not compute probabilities: {e}")

    st.subheader("🥇 Top 3 Best-Suited Crops")
    for name, score in zip(top3_labels, top3_scores):
        st.write(f"**{str(name).upper()}** — {round(float(score) * 100, 2)}% suitability")

    # ---------------------- WEATHER & SOIL ADVISORY ----------------------
    st.subheader("🌦️ Weather & Soil Advisory")

    if humidity > 80:
        st.info("💧 High humidity detected — good for rice, papaya, coconut.")
    if ph < 6:
        st.warning("⚠️ Soil is acidic — avoid crops like wheat; prefer tea, citrus, pineapple.")
    if ph > 8:
        st.warning("⚠️ Soil is highly alkaline — suitable for barley, cotton, and millets.")
    if temperature > 35:
        st.error("🌡️ Very hot climate — choose heat-tolerant crops like millet, sorghum, or groundnut.")
    if temperature < 15:
        st.info("❄️ Cool temperature detected — suitable for crops like peas, cabbage, and wheat.")
    if rainfall < 50:
        st.warning("🌧️ Very low rainfall — prefer drought-resistant crops like chickpea, bajra, or ragi.")
    if rainfall > 200:
        st.info("🌧️ Heavy rainfall — suitable for rice, jute, rubber, and sugarcane.")

    # ---------------------- INPUT FEATURE GRAPH ----------------------
    st.subheader("📊 Input Feature Distribution (Visualized)")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        features = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
        values = [N, P, K, temperature, humidity, ph, rainfall]
        ax.bar(features, values, color="#2E7D32")
        ax.set_ylabel("Value")
        ax.set_title("Soil & Climate Input Values")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render chart: {e}")

    # ----------------------  DOWNLOAD REPORT (safe) ----------------------
    st.subheader("📄 Download Prediction Report")

    # build report text using the safe top3 variables
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

    Top 3 Crop Suitability:
    1. {str(top3_labels[0]).upper()} — {top3_scores[0]:.4f}
    2. {str(top3_labels[1]).upper()} — {top3_scores[1]:.4f}
    3. {str(top3_labels[2]).upper()} — {top3_scores[2]:.4f}
    """

    try:
        buffer = io.BytesIO()
        buffer.write(report.encode())
        buffer.seek(0)
        st.download_button(
            label="📥 Download Report",
            data=buffer,
            file_name="crop_recommendation_report.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Download failed: {e}")


