# PlateScribe

PlateScribe is an advanced license plate detection model that identifies and extracts license plate numbers from images. The model uses the InceptionResNetV2 architecture for feature extraction and Optical Character Recognition (OCR) for text extraction. This README provides a comprehensive guide on how to set up and use the PlateScribe project.

## Project Overview

PlateScribe leverages state-of-the-art object detection techniques to locate license plates in images and employs OCR to extract text from the detected plates. The model is designed for high accuracy and efficiency in real-world scenarios.

## Quick Access

To quickly get started with PlateScribe, visit the GitHub repository at [PlateScribe GitHub Repository](https://github.com/yourusername/PlateScribe).

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/PlateScribe.git
   cd PlateScribe
   ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install Required Packages
    ```bash
    pip install -r requirements.txt
    ```

4. Download Pre-trained Model Make sure to download the pre-trained model file and place it in the `./output/ directory.` The model file is named `object_detection_e10.keras.`

5. Run the Application
    ```bash
    streamlit run app.py
    ```

## Components

For detailed descriptions of each module, functions, and their purposes, please visit the [PlateScribe Wiki](link).


## Architecture Diagram

```plaintext
+---------------------------+
|        XML Parser         |
|---------------------------|
| parse_xml_files()         |
| save_labels_to_csv()      |
| get_image_filename()      |
| extract_image_filenames() |
+---------------------------+
             |
             v
+---------------------------+
|       Bounding Box        |
|         Verifier          |
|---------------------------|
|   verify_bounding_box()   |
+---------------------------+
             |
             v
+---------------------------+
|      Data Processor       |
|---------------------------|
|      process_data()       |
+---------------------------+
             |
             v
+---------------------------+
|      Model Trainer        |
|---------------------------|
|  build_and_train_model()  |
+---------------------------+
             |
             v
+---------------------------+
|           App             |
|---------------------------|
| detect_objects_in_image() |
+---------------------------+
             |
             v
+---------------------------+
|   Streamlit Interface     |
|---------------------------|
|    User Uploads Image     |
|   Model Processes Image   |
| Bounding Boxes Displayed  |
| Extracted Text Displayed  |
+---------------------------+
```

## Flow of the Script
```plaintext
 Start
  |
  v
  XML Parser
  |
  v
Bounding Box Verifier
  |
  v
Data Processor
  |
  v
Model Trainer
  |
  v
  App
  |
  v
Streamlit Interface
```

## Conclusion

The PlateScribe project provides a full pipeline for detecting and reading license plates from images. It integrates multiple modules for parsing, verification, data processing, model training, and user interface to deliver a seamless experience for license plate detection and OCR.