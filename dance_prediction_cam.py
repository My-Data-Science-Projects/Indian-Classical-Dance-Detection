import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("dance_classifier_gpu.h5")

class_labels = ["Bharatanatyam", "Kathak", "Kathakali", "Kuchipudi", "Manipuri", "Mohiniyattam", "Odissi", "Sattriya"]

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0 
    frame = np.expand_dims(frame, axis=0) 
    return frame

st.title("Indian Classical Live Dance Detection")
st.write("Turn on the camera to predict the dance form in real-time!")

video_capture = cv2.VideoCapture(0)

FRAME_WINDOW = st.image([])

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to capture video")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    processed_frame = preprocess_frame(frame)

    predictions = model.predict(processed_frame)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    cv2.putText(frame, f"Prediction: {predicted_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (165, 42, 42), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(frame, use_column_width=True)

video_capture.release()
cv2.destroyAllWindows()
