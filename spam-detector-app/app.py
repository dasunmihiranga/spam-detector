import streamlit as st
import joblib
import pandas as pd
import os

# Get path relative to the location of this file
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '.', 'model', 'spam_detector.pkl')

# Load model
model = joblib.load(model_path)
# UI
st.title("📩 Spam Message Detector")

message = st.text_area("Enter your message")

if st.button("Check"):
    pred = model.predict([message])
    result = 'Spam ❌' if pred[0] == 1 else 'Ham ✅'
    st.success(f"Prediction: {result}")
