import cv2
import plotly.express as px
from skimage import io

def verify_bounding_box(image_path, xmin, ymin, xmax, ymax):
    img = io.imread(image_path)
    fig = px.imshow(img)
    fig.update_layout(
        width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),
        xaxis_title='Bounding Box Verification'
    )
    fig.add_shape(
        type='rect', x0=xmin, x1=xmax, y0=ymin, y1=ymax,
        xref='x', yref='y', line_color='cyan'
    )
    fig.show()