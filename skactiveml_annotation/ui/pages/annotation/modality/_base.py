from pathlib import Path
import logging

import pydantic

from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
)

import dash
from dash.exceptions import PreventUpdate

from skactiveml_annotation.hydra_schema.base import Modality
from skactiveml_annotation.ui.pages.annotation import actions

from . import (
    audio,
    image,
    text,
)
from ._model import DataDisplaySetting

from .. import (
    ids,
    actions,
)


def create_data_display(
    data_display_setting: DataDisplaySetting,
    modality: Modality,
    human_data_path: Path,
    dpr: float,
):
    w = dash.no_update
    h = dash.no_update

    if modality == Modality.IMAGE:
        image_display_setting = data_display_setting.image
        rendered_data, w, h = image.display(human_data_path, image_display_setting, dpr)
    elif modality == Modality.TEXT:
        text_display_setting = data_display_setting.text
        rendered_data = text.display(human_data_path, text_display_setting)
    elif modality == Modality.AUDIO:
        audio_display_setting = data_display_setting.audio
        rendered_data = audio.display(human_data_path, audio_display_setting)

    return (
        rendered_data,
        w,
        h,
    )

def create_data_presentation_settings(modality: Modality):
    if modality == Modality.IMAGE:
        return image.presentation_settings()
    elif modality == Modality.TEXT:
        return text.presentation_settings()
    elif modality == Modality.AUDIO:
        return audio.presentation_settings()


def register_callbacks(app: Dash):
    @app.callback(
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
        display_settings: DataDisplaySetting,
        checked_ids: list[dict[str, str]],
        checked_values: list[bool],
        ids: list[dict[str, str]],
        values: list[str | bool | int | float],
    ):
        if n_clicks is None:
            raise PreventUpdate

        _apply_updates(display_settings, checked_ids, checked_values)
        _apply_updates(display_settings, ids, values)

        return dict(
            ui_trigger=True,
            display_settings=display_settings,
        )
    _ = on_apply_data_presentation_settings


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
