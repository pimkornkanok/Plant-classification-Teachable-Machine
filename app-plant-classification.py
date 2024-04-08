
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Print title
st.title("Plant Classification App")

# Prepare the input image
input_img = st.file_uploader("Please upload an image to predict", type=['jpeg', 'jpg', 'png'])

if input_img is not None and st.button('Classify'):
    # Create the array of the right shape to feed into the Keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = Image.open(input_img).convert("RGB")
    
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    # turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predict the model
    prediction = model.predict(data)

    # Load the labels
    classes = [' '.join(x.split(' ')[1:]).replace('\n','') for x in open('labels.txt', 'r').readlines()]

    # Grab the labels from the labels.txt file. This will be used later.
    labels = [line.strip() for line in open('labels.txt', 'r').readlines()]

    # Display predictions

    
    prob = round(prediction[0,0] * 100, 2)
    st.write(f"Predicted Class: {classes[0]}, Confidence Score: {prob}%")
  
    prob = round(prediction[0,1] * 100, 2)
    st.write(f"Predicted Class: {classes[1]}, Confidence Score: {prob}%")
    
    prob = round(prediction[0,2] * 100, 2)
    st.write(f"Predicted Class: {classes[2]}, Confidence Score: {prob}%")
   
    prob = round(prediction[0,3] * 100, 2)
    st.write(f"Predicted Class: {classes[3]}, Confidence Score: {prob}%")


st.balloons()

