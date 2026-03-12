import streamlit as st
import numpy as np
from joblib import load
import plotly.graph_objects as go

# -------------------------------
# Load Model Files
# -------------------------------
model = load('model.joblib')
scaler = load('scaler.joblib')
encoder = load('encoder.joblib')

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(
    page_title="Carbon Emission Predictor",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🌱 Input Parameters")
country = st.sidebar.selectbox("Country", encoder.classes_)
year = st.sidebar.slider("Year", 1990, 2030, 2020)
population = st.sidebar.number_input("Population (Millions)", 1.0, 2000.0, 500.0)
gdp = st.sidebar.number_input("GDP (Billion USD)", 100.0, 50000.0, 10000.0)
energy = st.sidebar.number_input("Energy Consumption (TWh)", 100.0, 50000.0, 10000.0)
renewable = st.sidebar.slider("Renewable Energy %", 0.0, 100.0, 30.0)

# -------------------------------
# Main Header
# -------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#2E7D32;'>🌍 Carbon Emission Predictor</h1>"
    "<h4 style='text-align:center; color:gray;'>Sustainability ML Project</h4>",
    unsafe_allow_html=True
)
st.divider()

# -------------------------------
# Display Inputs as Metrics
# -------------------------------
st.subheader("📝 Overview of Inputs")
col1, col2, col3 = st.columns(3)

col1.metric("Country", country)
col1.metric("Year", year)

col2.metric("Population (M)", f"{population}")
col2.metric("GDP (B USD)", f"{gdp}")

col3.metric("Energy (TWh)", f"{energy}")
col3.metric("Renewable %", f"{renewable}%")

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Emission 🔮"):

    # Prepare data
    country_enc = encoder.transform([country])[0]
    data = np.array([[country_enc, year, population, gdp, energy, renewable]])
    data = scaler.transform(data)
    pred = model.predict(data)[0]

    # Prediction result text
    if pred == 1:
        st.markdown("<h2 style='color:red; text-align:center;'>🔴 High Carbon Emission Expected</h2>", unsafe_allow_html=True)
        score = 80  # Example risk score
    else:
        st.markdown("<h2 style='color:green; text-align:center;'>🟢 Low Carbon Emission Expected</h2>", unsafe_allow_html=True)
        score = 30  # Example risk score

    # -------------------------------
    # Visual Gauge
    # -------------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Carbon Emission Risk", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if score > 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#C8E6C9"},
                {'range': [50, 100], 'color': "#FFCDD2"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.divider()