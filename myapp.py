import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model
model = joblib.load("car_price_model.pkl")

st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered",
    page_icon="🚗",
)

# Sidebar
st.sidebar.title("Input Parameters")
st.sidebar.write("Fill in the car details below:")

year = st.sidebar.number_input("Year of Manufacture", 1990, 2024, 2018)
present_price = st.sidebar.number_input("Present Price (in Euros)", 0.1, 1000000.0, 2000.0)
kms = st.sidebar.number_input("Kilometers Driven", 0, 500000, 20000)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.number_input("Previous Owners", 0, 5, 0)

# One-hot encoding 
fuel_diesel = 1 if fuel == "Diesel" else 0
fuel_cng = 1 if fuel == "CNG" else 0
seller_individual = 1 if seller == "Individual" else 0
trans_auto = 1 if transmission == "Automatic" else 0

# Main Page
st.title("Car Price Prediction App")
st.subheader("Estimate the selling price of a car using ML")

st.write("---")
st.markdown("### Prediction Output")

if st.button("Predict Price"):
    input_data = np.array([
        year,
        present_price,
        kms,
        owner,
        fuel_diesel,
        fuel_cng,
        seller_individual,
        trans_auto
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: **{prediction:.2f} €**")

# Footer
st.write("---")
st.caption("Made using Streamlit | Fatin Kassabi")



