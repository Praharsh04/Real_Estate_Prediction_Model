import streamlit as st
import joblib
import pandas as pd

import os

# Get the absolute path to the directory containing this script
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(_APP_DIR, "model", "model.joblib")
PIPELINE_PATH = os.path.join(_APP_DIR, "model", "full_pipeline.joblib")

def load_model():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

def get_user_input():
    st.header("California Housing Price Prediction")
    longitude = st.number_input("Longitude", -124.0, -114.0, step=0.01)
    latitude = st.number_input("Latitude", 32.0, 42.0, step=0.01)
    housing_median_age = st.slider("Housing Median Age", 1, 52)
    total_rooms = st.number_input("Total Rooms", 2, 10000, step=1)
    total_bedrooms = st.number_input("Total Bedrooms", 1, 3000, step=1)
    population = st.number_input("Population", 1, 20000, step=1)
    households = st.number_input("Households", 1, 5000, step=1)
    median_income = st.number_input("Median Income", 0.0, 15.0, step=0.1)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )
    data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }
    return pd.DataFrame([data])

def main():
    model, pipeline = load_model()
    user_input = get_user_input()
    if st.button("Predict"):
        processed = pipeline.transform(user_input)
        prediction = model.predict(processed)[0]
        st.success(f"🏡 Predicted Median House Value: ${prediction * 100000:,.2f}")
        st.balloons()

if __name__ == "__main__":
    main()