import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.xml_parser import extract_image_filenames

def process_data(df: pd.DataFrame):
    data = []
    output = []
    
    labels = df.iloc[:, 1:5].values
    image_paths = extract_image_filenames(df)
    
    for index, image_path in enumerate(image_paths):
        image_array = cv.imread(image_path)
        height, width = image_array.shape[:2]
        
        # target_size being a constant helps all the image to be loaded with the same size
        # practically standardizing the shape of all the images
        load_image = load_img(image_path, target_size=(224,224))
        # this normalizes the image arrays to help to model convergence and performance
        # i.e. it makes the range from 0 to 255 into the range of 0 to 1
        load_image_array = img_to_array(load_image) / 255.0
        
        xmin, xmax, ymin, ymax = labels[index]
        normalized_labels = (
            xmin/width, 
            xmax/width, 
            ymin/height, 
            ymax/height
        )
        
        data.append(load_image_array)
        output.append(normalized_labels)
        
    independent_variable = np.array(data, dtype=np.float32)
    dependent_variable = np.array(data, dtype=np.float32)
    
    return train_test_split(
        independent_variable, 
        dependent_variable, 
        test_size=0.2, 
        random_state=0
    )
        
        