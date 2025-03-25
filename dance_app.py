import streamlit as st
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model("models/best_model.h5")

class_labels = ["Bharatanatyam", "Kathak", "Kathakali", "Kuchipudi", "Manipuri", "Mohiniyattam", "Odissi", "Sattriya"]

TEST_PATH = "dataset/test"

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Indian Classical Dance Detection")
st.write("Upload an image to classify the Indian dance form or select a test image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if os.path.exists(TEST_PATH):
    test_images = os.listdir(TEST_PATH)
    if test_images:
        selected_test_image = st.selectbox("Or select an image from test dataset:", test_images)
        test_image_path = os.path.join(TEST_PATH, selected_test_image)
    else:
        test_image_path = None
else:
    test_image_path = None

if uploaded_file is not None or test_image_path:
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(test_image_path)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    with tf.device('/GPU:0'):
        predictions = model.predict(processed_image)

    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # st.subheader(f"Prediction: **{predicted_class}**")
    # st.write(f"Confidence: **{confidence:.2f}**")
    st.markdown(
    f'<div style="background-color:#28a745; padding:10px; border-radius:5px; text-align:center;">'
    f'<h3 style="color:#ffffff;">Predicted Dance : <b style="font-style: italic;">{predicted_class}</b></h3></div>',
    unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with open("dance_description.json", "r") as file:
        benefits = json.load(file)

    st.markdown(
        """
        <style>
        .stExpander p {
            font-size: 24px;
        }
        .st-emotion-cache-1pbsqtx {  
            height: 2.25rem;
        }
        .stElementContainer p {
            font-size: 20px;
        }
        .st-emotion-cache-4rp1ik:hover {
            color: rgb(40, 167, 69);
        }
        details summary:hover svg {
            fill: green !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"**Benefits of {predicted_class} Dance**"):
        st.write(benefits.get(predicted_class, "No information available."))