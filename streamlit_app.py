# crop_recommender_streamlit.py
import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Page configuration and styling
# -------------------------------
st.set_page_config(page_title="Crop Recommender App", page_icon="🌾", layout="centered")

st.markdown(
    """
    <style>
    body {background-color: #EFD8F4;}
    .title {
        color: #264653;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🌾 Helping Farmers Choose Crops Using AI 🌾</div>', unsafe_allow_html=True)

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = r"Crop-Recommender.pkl"
    model = joblib.load(model_path)
    return model

model = load_model()

# -------------------------------
# Input fields
# -------------------------------
st.subheader("Enter Soil and Weather Details:")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Ratio of Nitrogen content in soil", min_value=0.0, step=0.1)
    phosphorus = st.number_input("Ratio of Phosphorous content in soil", min_value=0.0, step=0.1)
    potassium = st.number_input("Ratio of Potassium content in soil", min_value=0.0, step=0.1)
    ph = st.number_input("pH value of the soil", min_value=0.0, step=0.1)

with col2:
    temperature = st.number_input("Temperature in °C", min_value=0.0, step=0.1)
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Crop"):
    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    try:
        prediction = model.predict(features)
        crop = prediction[0]
        st.success(f"🌱 The recommended crop is **{crop}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Jayendrajeet Chauhan | Crop Recommendation using AI 🌾")
