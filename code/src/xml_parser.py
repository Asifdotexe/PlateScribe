import os
import xml.etree.ElementTree as xet
import pandas as pd
from glob import glob

def parse_xml_files(path_pattern: str) -> dict:
    """
    Parses XML files matching the provided path pattern and extracts bounding box information.

    Args:
    path_pattern (str): A string pattern to match XML file paths.

    Returns:
    dict: A dictionary containing the parsed XML files' information. The dictionary has the following keys:
        - 'filepath': A list of parsed XML file paths.
        - 'xmin': A list of x-coordinates of the bounding boxes.
        - 'xmax': A list of x-coordinates of the bounding boxes.
        - 'ymin': A list of y-coordinates of the bounding boxes.
        - 'ymax': A list of y-coordinates of the bounding boxes.
    """
    label_dictionary = {
        'filepath': [],
        'xmin': [],
        'xmax': [],
        'ymin': [],
        'ymax': []
    }
    for filename in glob(path_pattern):
        info = xet.parse(filename)
        root = info.getroot()
        member_object = root.find('object')
        labels_info = member_object.find('bndbox')

        xmin = int(labels_info.find('xmin').text)
        xmax = int(labels_info.find('xmax').text)
        ymin = int(labels_info.find('ymin').text)
        ymax = int(labels_info.find('ymax').text)

        label_dictionary['filepath'].append(filename)
        label_dictionary['xmin'].append(xmin)
        label_dictionary['xmax'].append(xmax)
        label_dictionary['ymin'].append(ymin)
        label_dictionary['ymax'].append(ymax)

    return label_dictionary

def save_labels_to_csv(label_dictionary: dict, csv_path: str) -> pd.DataFrame:
    """
    Saves the parsed XML files' information to a CSV file.

    Args:
    label_dictionary (dict): A dictionary containing the parsed XML files' information. The dictionary has the following keys:
        - 'filepath': A list of parsed XML file paths.
        - 'xmin': A list of x-coordinates of the bounding boxes.
        - 'xmax': A list of x-coordinates of the bounding boxes.
        - 'ymin': A list of y-coordinates of the bounding boxes.
        - 'ymax': A list of y-coordinates of the bounding boxes.
    csv_path (str): The path to save the CSV file.

    Returns:
    pd.DataFrame: A DataFrame object containing the parsed XML files' information saved to a CSV file.
    """
    df = pd.DataFrame(label_dictionary)
    df.to_csv(csv_path, index=False)
    return df