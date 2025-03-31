from dash import dcc
import dash_mantine_components as dmc

import plotly.express as px
from PIL import Image


# TODO make components out of these.
def create_image_display(path_to_img):
    # Use a separate Callback to update the image.
    image = Image.open(path_to_img).convert("RGB")
    print("Size of Image:")
    print(image.size)

    from PIL import ImageFilter
    from PIL.Image import Resampling

    # image = image.filter(ImageFilter.SMOOTH_MORE)

    w_original = image.width
    h_original = image.height

    factor = 10
    image = image.resize(
        (image.width * factor, image.height * factor),
        resample=Resampling.LANCZOS,
        # reducing_gap=4
    )

    # image = image.resize(w_original, h_original)

    fig = px.imshow(image, labels={})

    # INFO Graph does not support loading wrapper
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