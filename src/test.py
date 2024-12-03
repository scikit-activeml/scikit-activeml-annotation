from dash import Dash, html
import base64
from PIL import Image

"""
Examples of different methods of adding local images to your Dash App
Note - Recommended to keep image files inside assets folder
- app.py
- assets/
    |-- my-image.png
"""

#Using direct image file path
image_path = 'assets/my-image.png'

#Using Pillow to read the the image
pil_img = Image.open("assets/my-image.png")

# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


app = Dash(__name__)

app.layout = html.Div([
    html.H1('Dash Puppies'),

    html.Img(src=image_path),                          # passing the direct file path
    html.Img(src=app.get_asset_url('my-image.png')),    # usign get_asset_url function
   #html.Img(src=dash.get_asset_url('my-image.png'))    Or with newer Dash v2.2.0
    html.Img(src=pil_img),                             # using the pillow image variable
    html.Img(src=b64_image(image_path)),               # using base64 to encode and decode the image file
])

if __name__ == "__main__":
    app.run_server()