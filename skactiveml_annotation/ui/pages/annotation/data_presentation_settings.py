from dash import (
    Input,
    Output,
    State,
    callback,
)

from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from PIL.Image import Resampling as PIL_Resampling

from . import ids
from skactiveml_annotation.core.schema import (
    DataType,
)

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


@callback(
    Input(ids.RESAMPLING_FACTOR_INPUT, 'value'),
    Input(ids.RESAMPLING_METHOD_RADIO, 'value'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    output=dict(
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_image_presentation_settings_changed(
    rescale_factor,
    resampling_method,
    display_settings,
):
    display_settings = DataDisplaySetting.model_validate(display_settings)
    image_settings = display_settings.image
    image_settings.rescale_factor = rescale_factor
    image_settings.resampling_method = resampling_method

    print("Data Display Setting after confirm:")

    return dict(
        display_settings=display_settings.model_dump()
    )

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


@callback(
    Input(ids.FONT_SIZE_INPUT, 'value'),
    Input(ids.LINE_HEIGHT_INPUT, 'value'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    output=dict(
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_text_presentation_settings_changed(
    font_size,
    line_height,
    display_settings_json,
):
    display_settings = DataDisplaySetting.model_validate(display_settings_json)
    text_settings = display_settings.text
    
    text_settings.font_size = font_size
    text_settings.line_height = line_height

    return dict(
        display_settings=display_settings.model_dump()
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
    Input(ids.LOOP_INPUT, 'checked'),
    Input(ids.PLAYBACK_RATE_INPUT, 'value'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    output=dict(
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_audio_presentation_settings_changed(
    should_loop,
    playback_rate,
    # Data
    display_settings,
):
    display_settings = DataDisplaySetting.model_validate(display_settings)
    audio_settings = display_settings.audio

    audio_settings.loop = should_loop
    audio_settings.playback_rate = playback_rate

    return dict(
        display_settings=display_settings.model_dump()
    )


# Generic
@callback(
    Input("refresh-ui-button", 'n_clicks'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_refresh_ui_clicked(
    clicks
):
    if clicks is None:
        raise PreventUpdate

    return dict(
        ui_trigger=True,
    )
