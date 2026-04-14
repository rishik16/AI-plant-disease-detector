import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("model.h5")

# -------------------------------
# LOAD LABELS
# -------------------------------
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -------------------------------
# TREATMENT DICTIONARY
# -------------------------------
treatments = {
    "Tomato___Early_blight": "Remove infected leaves. Use fungicides like chlorothalonil or copper-based sprays.",
    "Tomato___Late_blight": "Use fungicides and avoid overhead watering. Remove infected plants immediately.",
    "Tomato___Leaf_Mold": "Ensure proper ventilation and apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove affected leaves and apply fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use neem oil or insecticidal soap.",
    "Tomato___Target_Spot": "Apply fungicide and avoid wet leaves.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Control whiteflies using insecticides. Remove infected plants.",
    "Tomato___mosaic_virus": "Remove infected plants. No chemical cure available.",
    "Tomato___healthy": "Your plant is healthy. No treatment needed.",

    "Potato___Early_blight": "Use fungicide and crop rotation.",
    "Potato___Late_blight": "Destroy infected plants and apply fungicides.",
    "Potato___healthy": "Healthy plant. Maintain good care.",

    "Pepper__bell___Bacterial_spot": "Use copper-based bactericides and avoid overhead irrigation.",
    "Pepper__bell___healthy": "Healthy plant. No action needed."
}

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌿 AI Plant Disease Detector")
st.write("Upload a leaf image to detect disease and get treatment.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Updated (no deprecated parameter)
    st.image(image, caption="Uploaded Image", width="stretch")

    st.write("🔍 Detecting disease...")

    # Prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    clean_name = predicted_class.replace("___", " - ")

    st.success(f"✅ Disease: {clean_name}")
    st.info(f"📊 Confidence: {confidence:.2f}")

    # -------------------------------
    # TREATMENT OUTPUT
    # -------------------------------
    default_treatment = "Use natural pesticide like neem oil or organic spray and monitor plant regularly."

    treatment = treatments.get(predicted_class, default_treatment)

    st.subheader("💊 Recommended Treatment")
    st.write(treatment)
