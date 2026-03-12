import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = tf.keras.models.load_model("digit_model.h5")

st.title("Handwritten Digit Digitizer")

st.write("Draw a digit (0-9)")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        img = np.mean(img,axis=2)

        img = Image.fromarray(img.astype('uint8'),'L')

        img = img.resize((28,28))

        img = np.array(img)

        img = img/255.0

        img = img.reshape(1,28,28,1)

        prediction = model.predict(img)

        digit = np.argmax(prediction)

        confidence = np.max(prediction)

        st.write("Prediction:",digit)
        st.write("Confidence:",confidence)
