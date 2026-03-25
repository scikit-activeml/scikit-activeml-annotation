from pathlib import Path
from io import BytesIO

from dash import dcc
import dash_mantine_components as dmc

from plotly import graph_objects as go

from PIL import Image as pil_image
from PIL.Image import Resampling as PIL_Resampling
 
from ._model import ImageDataDisplaySetting
    
from skactiveml_annotation.core import api

from .. import ids


IMAGE_RESAMPLING_METHOD_INPUT = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'image', 'index': 'resampling_method' }
IMAGE_RESIZING_FACTOR_INPUT = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'image', 'index': 'rescale_factor' }


def display(
    path_to_img: Path,
    image_display_setting: ImageDataDisplaySetting,
    dpr: float,
):
    image = pil_image.open(path_to_img).convert("RGB")

    rescale_factor = image_display_setting.rescale_factor

    image_widht = int(image.width * rescale_factor)
    image_height = int(image.height * rescale_factor)

    image = image.resize(
        (image_widht, image_height),
        resample=image_display_setting.resampling_method
    )

    # Account for screen dpr to prevent the browser from resizing the image again to avoid artifacts.
    layout_widht = int(image_widht / dpr)
    layout_height = int(image_height / dpr)

    ml = 0
    mt = 0
    mr = 0
    mb = 0
    fig = go.Figure(
        data=go.Image(
            source=_pil_image_to_inline_data_url(image, format="PNG"),
            # z=image,
        ),
        layout=go.Layout(
            width=max(10, layout_widht),
            height=max(10, layout_height),
            margin=dict(l=ml, r=mr, t=mt, b=mb, pad=0)
        )
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(autosize=False)

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
        layout_widht,
        layout_height,
    )


def presentation_settings():
    default_image_setting = ImageDataDisplaySetting()

    return (
        dmc.Stack(
            [
                dmc.NumberInput(
                    id=IMAGE_RESIZING_FACTOR_INPUT,
                    min=0.25,
                    max=50,
                    clampBehavior='strict',
                    hideControls=False,
                    step=0.5,
                    decimalScale=2,
                    label="Image resizing factor",
                    placeholder="1.0",
                    value=default_image_setting.rescale_factor,
                    allowNegative=False,
                    # w='35%',
                    persistence='resizing-factor-persistence',
                    persistence_type='session'
                ),

                dmc.RadioGroup(
                    dmc.Stack(
                        [
                            dmc.Radio(label='Nearest', value=str(PIL_Resampling.NEAREST)),
                            dmc.Radio(label='Lanczos', value=str(PIL_Resampling.LANCZOS)),
                        ],
                        align='start',
                        gap=5,
                    ),
                    persistence='resampling-method-persistence',
                    persistence_type='session',
                    label='Resampling Method',
                    # description="Choose method",
                    id=IMAGE_RESAMPLING_METHOD_INPUT,
                    value=str(default_image_setting.resampling_method),
                    size="sm"
                ),
            ],
            align='start'
        )
    )


def _pil_image_to_inline_data_url(pil_image: pil_image.Image, format: str ="PNG") -> str:
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        img:        PIL.Image.Image instance to encode.
        fmt:        Format to save the image in (e.g. "PNG", "JPEG").

    Returns:
        Data URL for the base64 encoded image.
    """
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    mime = f"image/{format.lower()}"
    return api.file_buffer_to_inline_data_url(buffer, mime)
