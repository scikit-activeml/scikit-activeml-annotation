import base64
from pathlib import Path
from io import BytesIO

from dash import dcc
import dash_mantine_components as dmc

from plotly import graph_objects as go

from PIL import Image as pil_image
 
from skactiveml_annotation.ui import data_display_settings
from skactiveml_annotation.ui.storekey import DataDisplayCfgKey

DEFAULT_RESIZE_FACTOR = 1
DEFAULT_RESAMPLING_METHOD = pil_image.Resampling.NEAREST


# TODO make components out of these.
def create_image_display(path_to_img: Path, dpr):
    image = pil_image.open(path_to_img).convert("RGB")

    # TODO set default values somewhere else.
    display_cfg = data_display_settings.get_or_default(
        'image',
        {
            DataDisplayCfgKey.RESCALE_FACTOR.value: DEFAULT_RESIZE_FACTOR,
            DataDisplayCfgKey.RESAMPLING_METHOD.value: DEFAULT_RESAMPLING_METHOD
        }
    )

    if not isinstance(display_cfg, dict):
        # TODO: enhance this error msg
        raise ValueError("Could not parse")

    factor = display_cfg[DataDisplayCfgKey.RESCALE_FACTOR.value]
    w = int(image.width * factor)
    h = int(image.height * factor)

    image = image.resize(
        (w, h),
        resample=display_cfg[DataDisplayCfgKey.RESAMPLING_METHOD.value]
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

    # Account for screen dpr to avoid resizing the image again to prevent artifacts.
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


def create_text_display(text) -> tuple[dmc.Container, int, int]:
    return (
        dmc.Container(
            dmc.Stack(
                dcc.Markdown(
                    text,  # Provide your text data here
                    className="markdown-content",
                    # Add additional Markdown options if necessary
                ),
            ),
        ),
        # TODO:
        10,
        10,
    )


def create_audio_display(audio) -> tuple[object, int, int]:
    print(audio)
    raise NotImplementedError
