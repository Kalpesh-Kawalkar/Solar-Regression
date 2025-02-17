

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("solar_power_model_xgb_best.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("☀️ Solar Power Generation Prediction")

st.sidebar.header("Adjust Inputs")
# Input fields
distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon (radians)", min_value=0.0, max_value=1.0, value=0.5)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10, max_value=50, value=25)
wind_direction = st.sidebar.number_input("Wind Direction (°)", min_value=0, max_value=360, value=180)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=3.5)
sky_cover = st.sidebar.selectbox("Sky Cover (0-4)", [0, 1, 2, 3, 4], index=2)
visibility = st.sidebar.number_input("Visibility (km)", min_value=0.0, max_value=50.0, value=10.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
average_wind_speed = st.sidebar.number_input("Average Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=3.0)
average_pressure = st.sidebar.number_input("Average Pressure (inHg)", min_value=28.0, max_value=32.0, value=29.5)

# Collect data for visualization
if "history" not in st.session_state:
    st.session_state.history = []

# Prediction
if st.button("🔍 Predict Power Generated"):
    input_data = pd.DataFrame([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                sky_cover, visibility, humidity, average_wind_speed, average_pressure]],
                              columns=["distance-to-solar-noon", "temperature", "wind-direction",
                                       "wind-speed", "sky-cover", "visibility", "humidity",
                                       "average-wind-speed-(period)", "average-pressure-(period)"])
    
    prediction = model.predict(input_data)[0]
    st.success(f"⚡ Predicted Power Generated: {prediction:.2f} Joules")

    # Store for history plot
    st.session_state.history.append(prediction)

    # **1️⃣ Show prediction trend**
    st.subheader("📊 Prediction Trend Over Time")
    if len(st.session_state.history) > 1:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(st.session_state.history) + 1), st.session_state.history, marker='o', linestyle='-', color='b')
        ax.set_xlabel("Prediction Count")
        ax.set_ylabel("Power Generated (Joules)")
        ax.set_title("Trend of Predictions Over Time")
        st.pyplot(fig)

    # **2️⃣ Show feature impact**
    st.subheader("🔍 Feature Impact on Prediction")
    feature_importance = np.random.rand(9)  # Placeholder (replace with actual feature importance if available)
    feature_names = input_data.columns
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, ax=ax, palette="coolwarm")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Impact of Features on Power Prediction")
    st.pyplot(fig)

    # **3️⃣ Heatmap of correlations**
    st.subheader("📈 Feature Correlation Heatmap")
    sample_data = pd.DataFrame(np.random.rand(20, 9), columns=input_data.columns)  # Placeholder sample data
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sample_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
