import plotly.express as px
from skimage import io

import plotly.express as px
from skimage import io

def verify_bounding_box(
    image_path: str,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int
) -> None:
    """
    Verifies a given bounding box on the specified image.

    Parameters:
    - image_path (str): The path to the image file.
    - xmin (int): The x-coordinate of the top-left corner of the bounding box.
    - ymin (int): The y-coordinate of the top-left corner of the bounding box.
    - xmax (int): The x-coordinate of the bottom-right corner of the bounding box.
    - ymax (int): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
    None: The function displays the image with the specified bounding box overlaid on it.
    """
    img = io.imread(image_path)
    fig = px.imshow(img)

    fig.update_layout(
        width=600,
        height=500,
        margin=dict(l=10, r=10, b=10, t=10),
        xaxis_title='Bounding Box Verification'
    )

    fig.add_shape(
        type='rect',
        x0=xmin,
        x1=xmax,
        y0=ymin,
        y1=ymax,
        xref='x',
        yref='y',
        line_color='cyan'
    )

    fig.show()