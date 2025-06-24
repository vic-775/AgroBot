import gdown
import tensorflow as tf
import os
import streamlit as st
import numpy as np
from PIL import Image
import inspect

# Loading the trained model
def get_model():
    model_path = "cnn_model_plant_disease.keras"
    file_id = "1-JiqmEC-EtOhbEAlDPnSn6Ag-fsv1XAy"
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    
    print("Loading model from:", inspect.getfile(tf.keras.models.load_model))

    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

# Model Prediction
def model_prediction(test_image):
    cnn_model = get_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    
    # converting image to array form
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    # converting input array into numpy / batch
    input_arr = np.array([input_arr])
    
    # predicting
    predictions = cnn_model.predict(input_arr)
    
    # return index of max element
    return np.argmax(predictions)