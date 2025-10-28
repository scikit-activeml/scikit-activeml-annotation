from dash import (
    Input,
    Output,
    State,
    callback,
    ALL,
    MATCH,
    ctx
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


# Generic for all data types
def create_data_display_modal():
    # TODO need to check which data type it is.
    return \
        dmc.Modal(
            title='Configure Data display',
            id=ids.DATA_DISPLAY_MODAL,
            centered=True,
            shadow='xl'
            # opened=True,
        )

def create_modal_content(data_type: DataType):
    if data_type == DataType.IMAGE:
        return image_modal()
    elif data_type == DataType.TEXT:
        return text_modal()
    else:
        return audio_modal()

# Open data display configuration modal
@callback(
    Input(ids.DATA_DISPLAY_BTN, 'n_clicks'),
    output=dict(
        show_modal=Output(ids.DATA_DISPLAY_MODAL, 'opened', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def show_data_display_modal(
        clicks
):
    if clicks is None:
        raise PreventUpdate

    return dict(show_modal=True)

# Image
def image_modal():
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
                    hideControls=True,
                    decimalScale=2,
                    label="Image resizing factor",
                    placeholder="1.0",
                    value=default_image_setting.rescale_factor,
                    allowNegative=False,
                    w='35%',
                    persistence='resizing-factor-persistence',
                    persistence_type='session'
                ),

                dmc.RadioGroup(
                    dmc.Stack(
                        [
                            dmc.Radio(label='Nearest', value=str(PIL_Resampling.NEAREST)),
                            dmc.Radio(label='Lanczos', value=str(PIL_Resampling.LANCZOS)),
                        ],
                        align='center',
                        gap=5,
                    ),
                    persistence='resampling-method-persistence',
                    persistence_type='session',

                    # label='Resampling Method',
                    # description="Choose method",
                    # id={"type": "tweak", "dtype": "image", "prop": "resampling_method"},
                    id=ids.RESAMPLING_METHOD_RADIO,
                    value=str(default_image_setting.resampling_method),
                    size="sm"
                ),

                dmc.Center(
                    dmc.Button(
                        'Confirm',
                        id=ids.CONFIRM_DATA_DISPLAY_BTN,
                        # id={"type": "confirm-btn", "dtype": "image"},
                        color='dark',
                    ),
                    w='100%'
                )
            ],
            align='start'
        ),


# TODO must I repeat the confirm button for each modal type?
# Confirm modal selection and close modal
@callback(
    Input(ids.CONFIRM_DATA_DISPLAY_BTN, 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State(ids.RESAMPLING_FACTOR_INPUT, 'value'),
    State(ids.RESAMPLING_METHOD_RADIO, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        show_modal=Output(ids.DATA_DISPLAY_MODAL, 'opened', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_confirm_image_data_display_btn(
    clicks,
    display_settings,
    rescale_factor,
    resampling_method,
):
    if clicks is None:
        raise PreventUpdate

    # TODO: Hardcoded for image data

    display_settings = DataDisplaySetting.model_validate(display_settings)
    image_settings = display_settings.image
    image_settings.rescale_factor = rescale_factor
    image_settings.resampling_method = resampling_method

    print("Data Display Setting after confirm:")
    # print(display_settings)

    print("ON Confirm")

    return dict(
        ui_trigger=True,
        show_modal=False,
        display_settings=display_settings.model_dump()
    )

# Text
def text_modal():
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
                    w='35%',
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
                    w='35%',
                    persistence='line-height-persistence',
                    persistence_type='session'
                ),

                dmc.Center(
                    dmc.Button(
                        'Confirm',
                        id=ids.CONFIRM_TEXT_DISPLAY_BTN,
                        color='dark',
                    ),
                    w='100%'
                )
            ],
            align='start'
        )


@callback(
    Input(ids.CONFIRM_TEXT_DISPLAY_BTN, 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State(ids.FONT_SIZE_INPUT, 'value'),
    State(ids.LINE_HEIGHT_INPUT, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        show_modal=Output(ids.DATA_DISPLAY_MODAL, 'opened', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_confirm_text_data_display_btn(
    clicks,
    display_settings_json,
    font_size,
    line_height,
):
    if clicks is None:
        raise PreventUpdate

    # TODO: Hardcoded for image data

    display_settings = DataDisplaySetting.model_validate(display_settings_json)
    text_settings = display_settings.text
    
    text_settings.font_size = font_size
    text_settings.line_height = line_height

    return dict(
        ui_trigger=True,
        show_modal=False,
        display_settings=display_settings.model_dump()
    )

# Audio
def audio_modal():
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
                    w='35%',
                    persistence=ids.PLAYBACK_RATE_INPUT,
                    persistence_type='session'
                ),

                dmc.Center(
                    dmc.Button(
                        'Confirm',
                        id=ids.CONFIRM_AUDIO_DISPLAY_BTN,
                        color='dark',
                    ),
                    w='100%'
                )
            ],
            align='start'
        )


# TODO simply the process of adding new modals
# Perhaps using All in one components pattern (AIO)
@callback(
    Input(ids.CONFIRM_AUDIO_DISPLAY_BTN, 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State(ids.LOOP_INPUT, 'checked'),
    State(ids.PLAYBACK_RATE_INPUT, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        show_modal=Output(ids.DATA_DISPLAY_MODAL, 'opened', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_confirm_audio_data_display_btn(
    clicks,
    display_settings,
    should_loop,
    playback_rate,
):
    if clicks is None:
        raise PreventUpdate

    # TODO: Hardcoded for image data

    display_settings = DataDisplaySetting.model_validate(display_settings)
    audio_settings = display_settings.audio

    audio_settings.loop = should_loop
    audio_settings.playback_rate = playback_rate

    print("ON Confirm")

    return dict(
        ui_trigger=True,
        show_modal=False,
        display_settings=display_settings.model_dump()
    )


# TODO use pattern matching callblack instead?
# @callback(
#     Input({"type": "confirm-btn", "dtype": ALL}, "n_clicks"),
#     State({"type": "tweak", "dtype": ALL, "prop": ALL}, "value"),
#     State(ids.DATA_DISPLAY_CFG_DATA, "data"),
#     output=dict(
#         ui_trigger=Output(ids.UI_TRIGGER, "data", allow_duplicate=True),
#         show_modal=Output(ids.DATA_DISPLAY_MODAL, "opened", allow_duplicate=True),
#         data_display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, "data", allow_duplicate=True),
#     ),
#     prevent_initial_call=True,
# )
# def on_confirm_data_display_btn(confirm_clicks, tweak_values, data_display_settings):
#     # Determine which confirm button was pressed
#     triggered = ctx.triggered_id
#     if not triggered:
#         raise PreventUpdate
#
#     dtype = triggered["dtype"]
#     print('dtype')
#     print(dtype)
#
#     print("tweak values")
#     print(tweak_values)
#
#     # Load persisted config into model
#     data_display_settings = DataDisplaySetting.model_validate(data_display_settings)
#
#     # Use ctx.states_list to access tweak metadata
#     # ctx.states_list[0] corresponds to the pattern-matched tweak states
#     states = ctx.states_list[0]
#     print(ctx.states_list)
#
#     # Filter only tweaks matching this dtype
#     tweaks_for_dtype = {
#         s["id"]["prop"]: s["value"]
#         for s in states
#         if s["id"]["dtype"] == dtype
#     }
#
#     print(f"Confirm triggered for dtype={dtype}")
#     print("Collected tweaks:", tweaks_for_dtype)
#
#     # --- Apply updates dynamically ---
#     if hasattr(data_display_settings, dtype):
#         target = getattr(data_display_settings, dtype)
#         for prop, val in tweaks_for_dtype.items():
#             if hasattr(target, prop):
#                 setattr(target, prop, val)
#
#     print("Data Display Setting after confirm:")
#     print(data_display_settings)
#
#     return dict(
#         ui_trigger=True,
#         show_modal=False,
#         data_display_settings=data_display_settings.model_dump(),
#     )
