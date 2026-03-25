from enum import StrEnum, auto

from dash import (
    Dash,
    Input,
    Output,
)

from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from . import ids

LABEL_SETTING_MODAL = { 'type': 'modal', 'index': "LabelSettingsModal" }
AUTO_ANNOTATE_MODAL = { 'type': 'modal', 'index': "AutoAnnotateModal" }


class SortBySetting(StrEnum):
    yaml_order = auto()
    alphabet = auto()
    proba = auto()

    def _generate_next_value(self, name, _start, _count, _last_values) -> str:
        return name


def create_label_settings_modal():
    return dmc.Modal(
        dmc.Stack(
            [
                dmc.Switch(
                    "Show class probabilities",
                    checked=True,
                    id=ids.LABEL_SETTING_SHOW_PROBAS,
                    persistence='show-proba-persistence',
                    persistence_type='local',
                    pt="xs",
                ),

                dmc.RadioGroup(
                    dmc.Stack(
                        [
                            dmc.Radio(
                                label='alphabet',
                                value=SortBySetting.alphabet.value,
                                size='md',
                            ),
                            dmc.Radio(
                                label='predicted class proba',
                                value=SortBySetting.proba.value,
                                size='md',
                            ),
                            dmc.Radio(
                                label='yaml config order',
                                value=SortBySetting.yaml_order.value,
                                size='md',
                            )
                        ],
                        gap=5,
                    ),
                    id=ids.LABEL_SETTING_SORTBY,
                    deselectable=False,
                    persistence='label-setting-sortby-persistence',
                    persistence_type='local',
                    value=SortBySetting.yaml_order.value,
                    label='Sort by',
                    size='md',
                ),

                dmc.Center(
                    dmc.Button(
                        'Confirm',
                        id=ids.LABEL_SETTING_CONFIRM_BTN,
                        color='dark',
                    ),
                    w='100%'
                )
            ],
        ),
        withCloseButton=False,
        withOverlay=True,
        id=LABEL_SETTING_MODAL,
        title='Label settings',
        centered=True,
        shadow='xl',
    )


def register_callbacks(app: Dash):
    @app.callback(
        Input(ids.LABEL_SETTING_BTN, 'n_clicks'),
        output=dict(
            show_modal=Output(LABEL_SETTING_MODAL, 'opened', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def show_label_settings_modal(
        clicks
    ):
        if clicks is None:
            raise PreventUpdate

        return dict(
            show_modal=True
        )
    _ = show_label_settings_modal


    @app.callback(
        Input(ids.LABEL_SETTING_CONFIRM_BTN, 'n_clicks'),
        output=dict(
            ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
            show_modal=Output(LABEL_SETTING_MODAL, 'opened', allow_duplicate=True)
        ),
        prevent_initial_call=True
    )
    def on_confirm(
        clicks,
    ):
        if clicks is None:
            raise PreventUpdate

        return dict(
            ui_trigger=True,
            show_modal=False
        )
    _ = on_confirm

