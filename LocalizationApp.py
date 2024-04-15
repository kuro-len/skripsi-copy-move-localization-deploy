import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model,Model

# Function to load the model
@st.cache_resource
def load_model(path):
    # Load the model
    model = tf.keras.models.load_model(path)
    return model

# Model Path
model_path = "model_comofod_preptrained_vgg16_unet_01"

st.title("Copy-Move Forgery Localization App")

# When the input changes, the cached model will be used
uploaded_file = st.file_uploader("Choose an image...", type=["png"])

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((512, 512))
    # Convert image to array
    image_array = np.asarray(image)
    # Expand dimensions to match input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if uploaded_file is not None:
    # Display the uploaded image
    image_input = Image.open(uploaded_file)
    st.image(image_input, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image_input)

    # Predict the image
    model = tf.keras.models.load_model(model_path,custom_objects={'F1Score':tfa.metrics.F1Score(num_classes=1,average="micro",threshold=0.5) })
    output = model.predict(image_array).reshape(512,512)
    
    st.image(output, caption='Predicted Mask.', use_column_width=True)
    
