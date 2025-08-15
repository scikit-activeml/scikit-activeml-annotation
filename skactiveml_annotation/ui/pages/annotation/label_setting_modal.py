from dash import (
    Input,
    Output,
    callback
)

from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from . import ids

SORT_BY_ALPHABET = '0'
SORT_BY_PROBA = '1'


def create_label_settings_modal():
    return dmc.Modal(
        dmc.Stack(
            [
                dmc.Switch(
                    "Show class probabilities",
                    checked=True,
                    id=ids.LABEL_SETTING_SHOW_PROBAS,
                    persistence='show-proba-persistence',
                    persistence_type='local'
                ),

                dmc.RadioGroup(
                    dmc.Stack(
                        [
                            dmc.Radio(
                                label='alphabet',
                                value=SORT_BY_ALPHABET,
                                size='md',
                            ),
                            dmc.Radio(
                                label='predicted class proba',
                                value=SORT_BY_PROBA,
                                size='md',
                            ),
                        ],
                        gap=5,
                    ),
                    id=ids.LABEL_SETTING_SORTBY,
                    deselectable=True,
                    persistence='label-setting-sortby-persistence',
                    persistence_type='local',
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
        id=ids.LABEL_SETTING_MODAL,
        title='Label settings',
        centered=True,
        shadow='xl',
    )


@callback(
    Input(ids.LABEL_SETTING_BTN, 'n_clicks'),
    output=dict(
        show_modal=Output(ids.LABEL_SETTING_MODAL, 'opened', allow_duplicate=True)
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


@callback(
    Input(ids.LABEL_SETTING_CONFIRM_BTN, 'n_clicks'),
    output=dict(
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        show_modal=Output(ids.LABEL_SETTING_MODAL, 'opened', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_confirm(
    clicks
):
    if clicks is None:
        raise PreventUpdate

    return dict(
        ui_trigger=True,
        show_modal=False
    )

