import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# -------------------------------
# LOAD MODEL (TFLITE)
# -------------------------------
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# LOAD LABELS
# -------------------------------
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -------------------------------
# TREATMENTS
# -------------------------------
treatments = {
    "Tomato___Early_blight": "Remove infected leaves and apply fungicide.",
    "Tomato___Late_blight": "Use fungicide and avoid excess moisture.",
    "Tomato___healthy": "Plant is healthy.",
    "Potato___Early_blight": "Use fungicide and crop rotation.",
    "Potato___Late_blight": "Destroy infected plants.",
    "Potato___healthy": "Healthy plant."
}

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# -------------------------------
# PREDICT
# -------------------------------
def predict(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# -------------------------------
# UI
# -------------------------------
st.title("🌿 Plant Disease Detector")

file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    st.write("🔍 Detecting...")

    processed = preprocess(img)
    preds = predict(processed)

    class_id = np.argmax(preds)
    confidence = np.max(preds)

    disease = class_names[class_id]

    st.success(f"✅ Disease: {disease}")
    st.info(f"Confidence: {confidence:.2f}")

    # Treatment
    if disease in treatments:
        treatment = treatments[disease]
    else:
        treatment = "Use natural pesticide like neem oil and maintain hygiene."

    st.write("💊 Treatment:", treatment)
