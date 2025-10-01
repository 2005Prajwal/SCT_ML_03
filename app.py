# app.py
import streamlit as st
import cv2
import numpy as np
import joblib
import os

# -----------------------------
# Load trained model
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "cat_dog_svm.pkl")  # updated name

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="🐱🐶 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Cat vs Dog Classifier 🐾")
st.markdown("Upload one or more images and see if they are Cats or Dogs!")

# -----------------------------
# Initialize session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# -----------------------------
# File uploader
# -----------------------------
uploaded_files = st.file_uploader(
    "Choose image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -----------------------------
# Process uploaded files
# -----------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Skip already processed files
        if uploaded_file.name in st.session_state.processed_files:
            continue
        st.session_state.processed_files.add(uploaded_file.name)

        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"⚠️ Could not read file: {uploaded_file.name}")
            continue

        # Preprocess image
        img_resized = cv2.resize(img, (64, 64))
        img_flatten = img_resized.flatten().reshape(1, -1)

        # Predict label
        prediction = model.predict(img_flatten)[0]
        label = "Cat" if prediction == 0 else "Dog"
        emoji = "🐱" if label == "Cat" else "🐶"

        # LinearSVC does NOT provide probability
        confidence = "N/A"

        # Display side by side
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, channels="BGR", caption=f"Uploaded: {uploaded_file.name}")
        with col2:
            if label == "Cat":
                st.success(f"{emoji} Prediction: {label}")
            else:
                st.warning(f"{emoji} Prediction: {label}")

        # Add to history
        st.session_state.history.append((img, label, confidence))

# -----------------------------
# Display prediction history
# -----------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    for i, (hist_img, hist_label, hist_conf) in enumerate(st.session_state.history[::-1], 1):
        st.image(hist_img, channels="BGR", caption=f"{i}. {hist_label} ({hist_conf})")
