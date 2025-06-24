import gdown
import tensorflow as tf
import os
import streamlit as st
import numpy as np
from PIL import Image


# Loading the trained model
def get_model():
    model_path = "cnn_model_plant_disease.keras"
    file_id = "1-JiqmEC-EtOhbEAlDPnSn6Ag-fsv1XAy"
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

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

# The app
st.sidebar.title("DashBoard")
pages = st.sidebar.selectbox("Select Page", ["Home", "About the Project", "Predictions"])

# Home Page
if (pages == 'Home'):
    st.header('AI Plant Diagnosis (CNN)')
    home_image = (r'app foto.jpeg')
    home_img = Image.open(home_image)
    home_img = home_img.resize((600,200))
    st.image(home_img)
    
    st.write("""
        Welcome to the AI Plant Diagnosis system!
          
        This tool uses a Convolutional Neural Network (CNN) model to detect plant diseases from images of leaves. 
         
        Simply upload a photo of a leaf on the Prediction page and let the system analyze and identify potential issues.
          
        It's fast, intelligent, and designed to assist farmers and plant enthusiasts in maintaining healthy crops.
    """)

    
# About Page
if (pages == 'About the Project'):
    st.header('About the Project')
    st.write("""
        This project is an AI-powered plant diagnosis system built using a Convolutional Neural Network (CNN).
        It was trained on thousands of labeled images of plant leaves, both healthy and diseased, to accurately 
        identify various plant conditions. 
        
        By uploading a photo of a leaf (predictions page), the system can predict the specific 
        disease affecting the plant (if any), helping farmers, gardeners, and researchers detect issues early 
        and take the appropriate measures.
        
        The model supports diagnosis for multiple crops including tomato, potato, corn, and more. The application 
        was developed using TensorFlow and Streamlit, making it accessible through a simple and user-friendly interface.
    """)

# Predictions Page
if (pages == 'Predictions'):
    st.header("AI Plant Diagnosis (CNN)")
    test_image = st.file_uploader("Upload Image")
    if (st.button("Show Image")):
        img = Image.open(test_image)
        # img = img.resize((250,250))
        st.image(img)

    if (st.button("Predict")):
        st.write("The Prediction")
        result_index = model_prediction(test_image)
        labels = []
        with open(r"labels.txt") as f:
            content = f.readlines()
            for i in content:
                labels.append(i[:-1])
            st.success('Its {}'.format(labels[result_index]))