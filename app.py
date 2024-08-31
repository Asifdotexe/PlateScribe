import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image
import pytesseract as pt

# Load your pre-trained model
model = tf.keras.models.load_model(r'./output/object_detection_e10.keras')

def detect_objects_in_image(input_image):
    # Convert to array and preprocess
    rgb_image_array = np.array(input_image.convert('RGB'), dtype=np.uint8)  # Ensure image is in RGB
    image_height, image_width, _ = rgb_image_array.shape
    resized_image = cv.resize(rgb_image_array, (224, 224)) / 255.0
    reshaped_image = resized_image.reshape(1, 224, 224, 3)

    # Make predictions
    bounding_box_coordinates = model.predict(reshaped_image)

    # Denormalize the bounding box values
    denormalization_array = np.array([image_width, image_width, image_height, image_height])
    bounding_box_coordinates = bounding_box_coordinates * denormalization_array
    bounding_box_coordinates = bounding_box_coordinates.astype(np.int32)

    # Draw bounding box on the image
    xmin, xmax, ymin, ymax = bounding_box_coordinates[0]
    cv.rectangle(rgb_image_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    return rgb_image_array, bounding_box_coordinates

def main():
    st.title("Plate Paparazzi")

    st.write(
        "Welcome to Plate Paparazzi. Our model is designed to detect and highlight license plates in images. 📸🚗"
        " Upload an image, and the model will identify the license plate, outline it, and extract the text for you."
    )
    st.write(
        "The model utilizes object detection techniques to locate license plates and performs optical character recognition (OCR)"
        " to extract the text from the detected plates."
    )

    # Upload an image
    uploaded_image_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_image_file is not None:
        # Load the image
        uploaded_image = Image.open(uploaded_image_file)

        st.write("Processing...")

        # Display progress bar
        progress_bar = st.progress(0)

        # Object detection
        image_with_bounding_box, bounding_box_coords = detect_objects_in_image(uploaded_image)
        progress_bar.progress(50)

        # Display the processed image with bounding box
        st.image(image_with_bounding_box, caption='Processed Image', use_column_width=True)
        progress_bar.progress(75)

        # Extract region of interest
        xmin, xmax, ymin, ymax = bounding_box_coords[0]
        region_of_interest = np.array(uploaded_image)[ymin:ymax, xmin:xmax]

        # Display the cropped image
        st.image(region_of_interest, caption='Cropped Image', use_column_width=True)
        progress_bar.progress(90)

        # Extract text from ROI using pytesseract
        extracted_text = pt.image_to_string(region_of_interest)
        progress_bar.progress(100)

        # Display extracted text
        st.subheader("Extracted Text:")
        st.text(extracted_text)

if __name__ == "__main__":
    main()