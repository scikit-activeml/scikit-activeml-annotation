import base64
from pathlib import Path
from io import BytesIO

import dash
from dash import dcc
import dash_mantine_components as dmc

from plotly import graph_objects as go

from PIL import Image as pil_image
 
from skactiveml_annotation.core.data_display_model import (
    TextDataDisplaySetting,
    ImageDataDisplaySetting,
)

# TODO make components out of these.
def create_image_display(
    path_to_img: Path, 
    image_display_setting: ImageDataDisplaySetting, 
    dpr: float
):
    image = pil_image.open(path_to_img).convert("RGB")

    rescale_factor = image_display_setting.rescale_factor
    w = int(image.width * rescale_factor)
    h = int(image.height * rescale_factor)

    image = image.resize(
        (w, h),
        resample=image_display_setting.resampling_method
        # reducing_gap=4  # Only in effect when downscaling
    )

    # image = image.filter(ImageFilter.UnsharpMask(
    #     radius=200,  # Effects the size of the edges to be enhanced. How wide the edge rims become
    #                   # Smaller Radius enhances smaller-scale detail
    #     percent=100,  # Amount is listed as a percentage and controlls how much constrast is added at edges
    #     threshold=3  # Controls how far adjacent tonal values have to be before the filter does anything
    #     )
    # )
    # image = ImageEnhance.Sharpness(image).enhance(50)

    # image.show()

    # Account for screen dpr to prevent the browser from resizing the image again to avoid artifacts.
    w = int(w / dpr)
    h = int(h / dpr)

    ml = 0
    mt = 0
    mr = 0
    mb = 0
    fig = go.Figure(
        data=go.Image(
            source=pil_image_to_base64(image, fmt="PNG"),
            # z=image,
        ),
        layout=go.Layout(
            width=w,
            height=h,
            margin=dict(l=ml, r=mr, t=mt, b=mb, pad=0)
        )
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return (
        dcc.Graph(
            figure=fig,
            responsive=False,
            style={
                # 'border':'blue solid 2px',
                # imageRendering:'pixelated'
            },
            config={
                # autosizeable=False,
                'scrollZoom':True,
                'doubleClick':'reset',
                'displaylogo':False,
                'modeBarButtonsToRemove':['toImage'],
            },
        ),
        w,
        h
    )


def pil_image_to_base64(img: pil_image.Image, fmt: str = "PNG") -> str:
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        img:        PIL.Image.Image instance to encode.
        fmt:        Format to save the image in (e.g. "PNG", "JPEG").

    Returns:
        Data URL for the base64 encoded image.
    """
    buffered = BytesIO()
    img.save(buffered, format=fmt)
    img_bytes = buffered.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64_str}"


def create_text_display(path: Path, text_display_setting: TextDataDisplaySetting):
    if not path.exists():
        raise ValueError(f"Cannot load text data from path: {path}")

    text_data = path.read_text(encoding="utf-8").strip()

    return \
        dmc.ScrollArea(
            dmc.Box(
                dcc.Markdown(
                    text_data,
                    style={
                        "margin": "0",
                        "padding": "0",
                        "fontSize": f"{text_display_setting.font_size}px",
                        "fontFamily": "'Consolas', 'Courier New', 'Lucida Console', monospace",
                        "fontVariantLigatures": "none",  # disable ligatures

                        "lineHeight": text_display_setting.line_height,
                        # Preserve line breaks but wrap long lines
                        "whiteSpace": "pre-line",  # allow wrapping (default is normal)
                        "wordBreak": "normal",  # break long words if needed
                        "overflowWrap": "normal", # ensures text doesn't overflow
                        # Width of the Text Container
                        "overflowX": "hidden", # no horizontal scrollbar
                        
                        # 'border':'blue dotted 2px',
                    }
                ),
                py=1,
                style={
                    # TODO hardcoded width
                    "width": "46vw",        # restrict component width
                    "overflowX": "hidden",      # no horizontal scrollbar

                    # "border": "2px solid green",  # Debug border
                },
            ),
            type='auto',
            offsetScrollbars='y',
            styles=dict(
                viewport={
                    'maxHeight': '60vh',
                    # 'border':'brown dashed 3px',
                },
            )
        )


def create_audio_display(audio, audio_display_setting):
    print(audio)
    raise NotImplementedError
