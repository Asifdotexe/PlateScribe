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

### 1. Modules

#### `xml_parser.py`

- **Purpose**: Contains functions for parsing XML files and extracting bounding box information.
- **Functions**:
  - `parse_xml_files(path_pattern: str) -> dict`: Parses XML files to extract bounding box coordinates.
  - `save_labels_to_csv(label_dictionary: dict, csv_path: str) -> pd.DataFrame`: Saves extracted bounding box information to a CSV file.
  - `get_image_filename(xml_filename: str) -> str`: Extracts the image filename from an XML file.
  - `extract_image_filenames(df: pd.DataFrame) -> list`: Extracts image filenames from a DataFrame.

#### `bounding_box_verifier.py`

- **Purpose**: Provides functionality to verify bounding boxes on images.
- **Functions**:
  - `verify_bounding_box(image_path: str, xmin: int, ymin: int, xmax: int, ymax: int) -> None`: Displays an image with the specified bounding box overlaid.

#### `model_trainer.py`

- **Purpose**: Responsible for building and training the license plate detection model.
- **Functions**:
  - `build_and_train_model(x_train, y_train, x_test, y_test) -> Model`: Constructs and trains the neural network model using InceptionResNetV2.

#### `data_processor.py`

- **Purpose**: Handles data preprocessing, including image loading and splitting.
- **Functions**:
  - `process_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: Processes image data and splits it into training and testing sets.

#### `app.py`

- **Purpose**: The main application script using Streamlit for the user interface.
- **Functions**:
  - `detect_objects_in_image(input_image: Image) -> tuple`: Detects objects in the uploaded image and draws bounding boxes.

## Architecture Diagram

```plaintext
+-------------------+
|    XML Parser     |
|-------------------|
| parse_xml_files() |
| save_labels_to_csv() |
| get_image_filename() |
| extract_image_filenames() |
+-------------------+
           |
           v
+-------------------+
| Bounding Box      |
|     Verifier      |
|-------------------|
| verify_bounding_box() |
+-------------------+
           |
           v
+-------------------+
|  Data Processor   |
|-------------------|
| process_data()    |
+-------------------+
           |
           v
+-------------------+
|  Model Trainer    |
|-------------------|
| build_and_train_model() |
+-------------------+
           |
           v
+-------------------+
|        App        |
|-------------------|
| detect_objects_in_image() |
+-------------------+
           |
           v
+-------------------+
| Streamlit Interface |
|-------------------|
| User Uploads Image |
| Model Processes Image |
| Bounding Boxes Displayed |
| Extracted Text Displayed |
+-------------------+
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