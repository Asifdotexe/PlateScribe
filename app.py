import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image
import pytesseract as pt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import plotly.express as px

# Load your pre-trained model
model = tf.keras.models.load_model(r'./output/object_detection_e10.keras')

def object_detection(image):
    # Convert to array and preprocess
    image = np.array(image.convert('RGB'), dtype=np.uint8)  # Ensure image is in RGB
    h, w, _ = image.shape
    image_resized = cv.resize(image, (224, 224)) / 255.0
    test_arr = image_resized.reshape(1, 224, 224, 3)

    # Make predictions
    coords = model.predict(test_arr)

    # Denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)

    # Draw bounding box on the image
    xmin, xmax, ymin, ymax = coords[0]
    cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    return image, coords

def main():
    st.title("Object Detection and OCR Pipeline")
    
    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Object detection
        processed_image, coords = object_detection(image)
        
        # Display the processed image with bounding box
        st.image(processed_image, caption='Processed Image', use_column_width=True)
        
        # Extract region of interest
        xmin, xmax, ymin, ymax = coords[0]
        roi = np.array(image)[ymin:ymax, xmin:xmax]
        
        # Display the cropped image
        st.image(roi, caption='Cropped Image', use_column_width=True)
        
        # Extract text from ROI using pytesseract
        text = pt.image_to_string(roi)
        
        # Display extracted text
        st.subheader("Extracted Text:")
        st.text(text)

if __name__ == "__main__":
    main()