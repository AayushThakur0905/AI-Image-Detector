from tensorflow.keras.applications.efficientnet import preprocess_input
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")

st.title("AI vs Real Image Detector")
st.write("Upload an image to check whether it is AI-generated or Real.")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/efficientnet_clean.keras",
        compile=False
    )
    return model

model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
IMG_SIZE = 224

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    image = preprocess_input(image)  # IMPORTANT FIX

    image = np.expand_dims(image, axis=0)
    return image
# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    st.write("Raw prediction:", float(prediction))

    st.subheader("Prediction Result:")

    if prediction > 0.5:
        st.error(f"AI Generated Image ({prediction:.2f} confidence)")
    else:
        st.success(f"Real Image ({1 - prediction:.2f} confidence)")