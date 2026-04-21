import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/gesture_model.h5")
labels = ['peace', 'stop', 'thumbs_up']

st.title("Hand Gesture Identifier")

img_file = st.camera_input("Take a picture")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    prediction = model.predict(img)
    label = labels[np.argmax(prediction)]

    st.write("Prediction:", label)
    st.write("Confidence:", np.max(prediction))