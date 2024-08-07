import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained ResNet50 model
model = load_model('path/to/your/saved/model.h5')

# Define the class names
class_names = ['Real', 'Fake']

def preprocess_image(image):
    # Resize the image to 224x224
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Expand dimensions to match the input shape for the model
    image = np.expand_dims(image, axis=0)
    # Preprocess the image using ResNet50's preprocess_input function
    image = preprocess_input(image)
    return image

def main():
    st.title("Image Classification with ResNet50")

    # File uploader to upload images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make predictions
        predictions = model.predict(preprocessed_image)
        prediction_class = np.argmax(predictions, axis=1)

        # Display the prediction
        st.write(f"The image is classified as: {class_names[prediction_class[0]]}")

if __name__ == "__main__":
    main()
