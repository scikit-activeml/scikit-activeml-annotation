from dash import (
    ALL,
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
from skactiveml_annotation.ui.pages.annotation import actions
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


# Image
def image_presentation_settings():
    default_image_setting = ImageDataDisplaySetting()

    return \
        dmc.Stack(
            [
                dmc.NumberInput(
                    id=ids.IMAGE_RESIZING_FACTOR_INPUT,
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
                    id=ids.IMAGE_RESAMPLING_METHOD_INPUT,
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
                    id=ids.TEXT_FONT_SIZE_INPUT,
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
                    id=ids.TEXT_LINE_HEIGHT_INPUT,
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
                    id=ids.AUDIO_LOOP_INPUT,
                    label="Looping",
                    checked=default_audio_setting.loop,
                    persistence=ids.LOOP_INPUT,
                    persistence_type='session'
                ),
            
                dmc.NumberInput(
                    id=ids.AUDIO_PLAYBACK_RATE_INPUT,
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


@callback(
    Input(actions.APPLY.btn_id, 'n_clicks'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    # Pattern Matching ids
    State({'type': ids.DATA_PRESENTATION_INPUT, 'property': 'checked', 'modality': ALL, 'index': ALL}, 'id'),
    State({'type': ids.DATA_PRESENTATION_INPUT, 'property': 'checked', 'modality': ALL, 'index': ALL}, 'checked'),
    State({'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': ALL, 'index': ALL}, 'id'),
    State({'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': ALL, 'index': ALL}, 'value'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        display_settings=Output(ids.DATA_DISPLAY_CFG_DATA, 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True,
)
def on_apply_data_presentation_settings(
    n_clicks: int | None,
    display_settings_json: dict,
    checked_ids: list[dict[str, str]],
    checked_values: list[bool],
    value_ids: list[dict[str, str]],
    value_values: list[str | bool | int | float],
):
    if n_clicks is None:
        raise PreventUpdate

    display_settings = DataDisplaySetting.model_validate(display_settings_json)

    _apply_updates(display_settings, checked_ids, checked_values)
    _apply_updates(display_settings, value_ids, value_values)

    return dict(
        ui_trigger=True,
        display_settings=display_settings.model_dump()
    )


def _apply_updates(
    display_settings: DataDisplaySetting,
    ids: list[dict],
    values: list,
):
    for cid, val in zip(ids, values):
        # modality field will exist otherwise they the ids would not be matched
        # for the callback.
        modality = cid["modality"] # "audio", "image", "text"
        field = cid["index"] # "loop", "playback_rate", ...

        if not hasattr(display_settings, modality):
            logging.error(f"Unknown modality '{modality}' in id {cid!r}")
            raise PreventUpdate
        submodel = getattr(display_settings, modality)

        if not hasattr(submodel, field):
            logging.error(f"Unknown field '{field}' in id {cid!r}")
            raise PreventUpdate

        try:
            # Radio buttons use string as type ...
            setattr(submodel, field, val)
        except pydantic.ValidationError as e:
            logging.error(
                f"Invalid data presentation setting applied: %s",
                _format_pydantic_validation_error(e)
            )
            raise PreventUpdate


def _format_pydantic_validation_error(e: pydantic.ValidationError) -> str:
    err = e.errors()[0]
    field = err["loc"][0]
    msg = err["msg"]
    inp = err["input"]
    return f"{field} {msg} (got {inp!r})"
