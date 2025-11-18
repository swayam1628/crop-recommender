# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---- Helpers ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

def predict_single(sample, model, scaler, label_encoder):
    # sample: list or 2D array of shape (1,7)
    sample_scaled = scaler.transform(np.array(sample).reshape(1, -1))
    pred_idx = model.predict(sample_scaled)
    pred_label = label_encoder.inverse_transform(pred_idx)[0]
    return pred_label

# ---- Page config ----
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("Crop Recommendation System 🌾")
st.write("Enter soil & weather features to get a crop recommendation.")

# load
model, scaler, label_encoder = load_artifacts()

# Input UI
st.header("Input features")
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("N (Nitrogen)", min_value=0.0, value=90.0, step=1.0, format="%.2f")
    P = st.number_input("P (Phosphorus)", min_value=0.0, value=40.0, step=1.0, format="%.2f")
    K = st.number_input("K (Potassium)", min_value=0.0, value=43.0, step=1.0, format="%.2f")
with col2:
    temperature = st.number_input("Temperature (°C)", value=26.8, format="%.2f")
    humidity = st.number_input("Humidity (%)", value=72.4, format="%.2f")
    ph = st.number_input("pH", value=7.0, format="%.2f")
with col3:
    rainfall = st.number_input("Rainfall (mm)", value=189.9, format="%.2f")

# Action
if st.button("Recommend Crop"):
    sample = [N, P, K, temperature, humidity, ph, rainfall]
    try:
        pred_label = predict_single(sample, model, scaler, label_encoder)
        st.success(f"Recommended crop: **{pred_label}**")
    except Exception as e:
        st.error("Prediction failed. Check server logs.")
        st.error(e)

st.markdown("---")
st.header("Example")
st.write("Try the example below or change values above and click Recommend.")

example = {"N": 90, "P":42, "K":43, "temperature":20.8, "humidity":82.0, "ph":6.5, "rainfall":202.9}
st.write(pd.DataFrame([example]))

if st.button("Use Example Values"):
    sample = [example['N'], example['P'], example['K'], example['temperature'],
              example['humidity'], example['ph'], example['rainfall']]
    try:
        pred_label = predict_single(sample, model, scaler, label_encoder)
        st.success(f"Recommended crop (example): **{pred_label}**")
    except Exception as e:
        st.error("Prediction failed.")
        st.error(e)

st.markdown("---")
st.write("Made with :heart: — Crop Recommendation System")
