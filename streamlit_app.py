import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")  # adjust this if your model filename is different

# Load your dataset automatically
df = pd.read_csv("engineered_telco_churn.csv")  # replace with your CSV filename

# Make predictions
predictions = model.predict(df)

# Display
st.title("Customer Churn Prediction")
st.write("### Input Data")
st.write(df)

st.write("### Predictions")
st.write(predictions)
