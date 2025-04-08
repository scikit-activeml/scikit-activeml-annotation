from dash import dcc
import dash_mantine_components as dmc

import plotly.express as px
from PIL import Image
from PIL.Image import Resampling


# TODO make components out of these.
def create_image_display(path_to_img, factor, is_lanzos):
    image = Image.open(path_to_img).convert("RGB")

    if factor != '':
        new_width = int(image.width * factor)
        new_height = int(image.height * factor)
        image = image.resize(
            (new_width, new_height),
            resample=Resampling.LANCZOS,
        )

    fig = px.imshow(image, labels={})

    return (
        dcc.Graph(
            figure=fig,
            # responsive=True,
            style={"width": "100%", "height": "auto"}
            # style={"border": "blue solid 1px"}
        ),
    )


def create_text_display(text):
    return (
        dmc.Container(
            dmc.Stack(
                dcc.Markdown(
                    text,  # Provide your text data here
                    className="markdown-content",
                    # Add additional Markdown options if necessary
                ),
            ),
        )
    )


def create_audio_display(audio):
    print(audio)
    raise NotImplementedError