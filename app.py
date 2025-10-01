# app.py
import streamlit as st
import joblib
import os
from PIL import Image
import numpy as np

# -------------------------------
# Set up model path safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_svm.pkl")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Cats vs Dogs Classifier 🐱🐶")
st.write("Upload an image and the model will predict if it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using Pillow
    img = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess for model
    # Resize image to the size used during training
    img = img.resize((64, 64))  # adjust this to match your training
    img_array = np.array(img).flatten().reshape(1, -1)  # flatten to 1D if your SVM expects that

    # Make prediction
    prediction = model.predict(img_array)[0]
    
    st.write(f"Prediction: **{prediction.upper()}**")
