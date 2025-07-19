import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('model/spam_detector.pkl')

# UI
st.title("📩 Spam Message Detector")

message = st.text_area("Enter your message")

if st.button("Check"):
    pred = model.predict([message])
    result = 'Spam ❌' if pred[0] == 1 else 'Ham ✅'
    st.success(f"Prediction: {result}")
