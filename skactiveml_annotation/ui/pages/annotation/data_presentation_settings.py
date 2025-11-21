from dash import (
    Input,
    Output,
    State,
    callback,
)

from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from PIL.Image import Resampling as PIL_Resampling
import pydantic

from . import ids
from skactiveml_annotation.core.schema import (
    DataType,
)

from skactiveml_annotation.util import logging
from skactiveml_annotation.core.data_display_model import (
    DataDisplaySetting,
    ImageDataDisplaySetting,
    TextDataDisplaySetting,
    AudioDataDisplaySetting,
)


def create_data_presentation_settings(data_type: DataType):
    if data_type == DataType.IMAGE:
        return image_presentation_settings()
    elif data_type == DataType.TEXT:
        return text_presentation_settings()
    else:
        return audio_presentation_settings()


def create_apply_button(data_type: DataType):
    if data_type == DataType.IMAGE:
        return _create_apply_button("image-presentation-setting-apply-btn")
    elif data_type == DataType.TEXT:
        return _create_apply_button("text-presentation-setting-apply-btn")
    else:
        return _create_apply_button("audio-presentation-setting-apply-btn")


def _create_apply_button(button_id: str):
    return (
        dmc.Center(
            dmc.Tooltip(
                dmc.Button(
                    "Apply",
                    id=button_id,
                    color='dark'
                ),
                label="Apply Presentation Settings now"
            )
        )
    )


# Image
def image_presentation_settings():
    default_image_setting = ImageDataDisplaySetting()

    return \
        dmc.Stack(
            [
                dmc.NumberInput(
                    # id={"type": "tweak", "dtype": "image", "prop": "resampling_factor"},
                    id=ids.RESAMPLING_FACTOR_INPUT,
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
                    # id={"type": "tweak", "dtype": "image", "prop": "resampling_method"},
                    id=ids.RESAMPLING_METHOD_RADIO,
                    value=str(default_image_setting.resampling_method),
                    size="sm"
                ),
            ],
            align='start'
        )


# Text
def text_presentation_settings():
    default_text_setting = TextDataDisplaySetting()

    return \
        dmc.Stack(
            [
                dmc.NumberInput(
                    # id={"type": "tweak", "dtype": "image", "prop": "resampling_factor"},
                    id=ids.FONT_SIZE_INPUT,
                    min=1,
                    max=35,
                    step=1,
                    clampBehavior='strict',
                    hideControls=False,
                    decimalScale=2,
                    label="Font size",
                    placeholder=str(default_text_setting.font_size),
                    value=default_text_setting.font_size,
                    allowNegative=False,
                    # w='35%',
                    persistence='font-size-persistence',
                    persistence_type='session'
                ),
            
                dmc.NumberInput(
                    id=ids.LINE_HEIGHT_INPUT,
                    min=0.2,
                    max=35,
                    step=0.1,
                    clampBehavior='strict',
                    hideControls=False,
                    decimalScale=2,
                    label="Line height",
                    placeholder=str(default_text_setting.line_height),
                    value=default_text_setting.line_height,
                    allowNegative=False,
                    # w='35%',
                    persistence='line-height-persistence',
                    persistence_type='session'
                ),
            ],
            align='start'
        )


# Audio
def audio_presentation_settings():
    default_audio_setting = AudioDataDisplaySetting()

    return \
        dmc.Stack(
            [
                dmc.Checkbox(
                    id=ids.LOOP_INPUT,
                    label="Looping",
                    checked=default_audio_setting.loop,
                    persistence=ids.LOOP_INPUT,
                    persistence_type='session'
                ),
            
                dmc.NumberInput(
                    id=ids.PLAYBACK_RATE_INPUT,
                    min=0.2,
                    max=35,
                    step=0.1,
                    clampBehavior='strict',
                    hideControls=False,
                    decimalScale=2,
                    label="Playback Rate",
                    placeholder=str(default_audio_setting.playback_rate),
                    value=default_audio_setting.playback_rate,
                    allowNegative=False,
                    # w='35%',
                    persistence=ids.PLAYBACK_RATE_INPUT,
                    persistence_type='session'
                ),
            ],
            align='start'
        )


# TODO: Organize this new code it does not belong here:
@callback(
    Input("image-presentation-setting-apply-btn", 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State(ids.RESAMPLING_FACTOR_INPUT, 'value'),
    State(ids.RESAMPLING_METHOD_RADIO, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True,
    allow_missing=True
)
def apply_image_presentation_settings(
    apply_clicks: None | int,
    display_settings_json,
    # Settings
    rescale_factor: float,
    resampling_method: str,
):
    if apply_clicks is None:
        raise PreventUpdate

    display_settings = DataDisplaySetting.model_validate(display_settings_json)

    try:
        image_settings = display_settings.image
        image_settings.rescale_factor = rescale_factor
        # Convert string to enum value
        image_settings.resampling_method = int(resampling_method)
    except pydantic.ValidationError as e:
        logging.error(
            f"Invalid data presentation setting applied: %s",
            _format_pydantic_validation_error(e)
        )
        raise PreventUpdate

    return dict(
        ui_trigger=True,
        display_settings=display_settings.model_dump()
    )


@callback(
    Input("text-presentation-setting-apply-btn", 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State(ids.FONT_SIZE_INPUT, 'value'),
    State(ids.LINE_HEIGHT_INPUT, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def apply_text_presentation_settings(
    apply_clicks: None | int,
    display_settings_json,
    # Settings
    font_size: int,
    line_height: int,
):
    if apply_clicks is None:
        raise PreventUpdate

    display_settings = DataDisplaySetting.model_validate(display_settings_json)

    try:
        text_settings = display_settings.text
        text_settings.font_size = font_size
        text_settings.line_height = line_height
    except pydantic.ValidationError as e:
        logging.error(
            f"Invalid data presentation setting applied: %s",
            _format_pydantic_validation_error(e)
        )
        raise PreventUpdate

    return dict(
        ui_trigger=True,
        display_settings=display_settings.model_dump()
    )


@callback(
    Input("audio-presentation-setting-apply-btn", 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State(ids.LOOP_INPUT, 'checked'),
    State(ids.PLAYBACK_RATE_INPUT, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def apply_audio_presentation_settings(
    apply_clicks: None | int,
    display_settings_json,
    # Settings
    should_loop: bool,
    playback_rate: float
):
    if apply_clicks is None:
        raise PreventUpdate

    display_settings = DataDisplaySetting.model_validate(display_settings_json)

    try:
        audio_settings = display_settings.audio
        audio_settings.loop = should_loop
        audio_settings.playback_rate = playback_rate
    except pydantic.ValidationError as e:
        logging.error(
            f"Invalid data presentation setting applied: %s",
            _format_pydantic_validation_error(e)
        )
        raise PreventUpdate

    return dict(
        ui_trigger=True,
        display_settings=display_settings.model_dump()
    )


def _format_pydantic_validation_error(e: pydantic.ValidationError) -> str:
    err = e.errors()[0]
    field = err["loc"][0]
    msg = err["msg"]
    inp = err["input"]
    return f"{field} {msg} (got {inp!r})"
