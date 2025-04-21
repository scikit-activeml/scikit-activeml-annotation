
from dash import (
    Input,
    Output,
    State,
    callback
)

from dash.exceptions import PreventUpdate

from ui import cache_storage
from ui.pages.annotation.ids import *
import dash_mantine_components as dmc

from PIL.Image import Resampling

from ui.storekey import DataDisplayCfgKey


def create_data_display_modal():
    # TODO need to check which data type it is.

    return \
        dmc.Modal(
            [
                dmc.Stack(
                    [
                        dmc.NumberInput(
                            id=RESAMPLING_FACTOR_INPUT,
                            min=0.25,
                            max=50,
                            clampBehavior='strict',
                            hideControls=True,
                            decimalScale=2,
                            label="Image resizing factor",
                            placeholder="1.0",
                            # value=1.0,
                            allowNegative=False,
                            w='35%',
                            persistence='resizing-factor-persistence',
                            persistence_type='local'
                       ),

                        dmc.RadioGroup(
                            dmc.Stack(
                                [
                                    dmc.Radio(label='Nearest', value=str(Resampling.NEAREST)),
                                    dmc.Radio(label='Lanczos', value=str(Resampling.LANCZOS)),
                                ],
                                align='center',
                                gap=5,
                            ),
                            persistence='resampling-method-persistence',
                            persistence_type='local',

                            # label='Resampling Method',
                            # description="Choose method",
                            id=RESAMPLING_METHOD_RADIO,
                            value=str(Resampling.NEAREST),
                            size="sm"
                        ),

                        dmc.Center(
                            dmc.Button(
                                'Confirm',
                                id=CONFIRM_DATA_DISPLAY_BTN,
                                color='dark',
                            ),
                            w='100%'
                        )
                    ],
                    align='start'
                )
            ],
            title='Configure Data display',
            id=DATA_DISPLAY_MODAL,
            centered=True,
            shadow='xl'
            # opened=True,
        )


# Open data display configuration modal
@callback(
    Input(DATA_DISPLAY_BTN, 'n_clicks'),
    output=dict(
        show_modal=Output(DATA_DISPLAY_MODAL, 'opened', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def show_data_display_modal(
        clicks
):
    if clicks is None:
        raise PreventUpdate

    return dict(show_modal=True)


# Confirm modal selection and close modal
@callback(
    Input(CONFIRM_DATA_DISPLAY_BTN, 'n_clicks'),
    State(RESAMPLING_FACTOR_INPUT, 'value'),
    State(RESAMPLING_METHOD_RADIO, 'value'),
    output=dict(
        ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True),
        show_modal=Output(DATA_DISPLAY_MODAL, 'opened', allow_duplicate=True),
        # data_display_store=Output(DATA_DISPLAY_CFG_STORE, 'data')
    ),
    prevent_initial_call=True
)
def on_confirm_data_display_btn(
        clicks,
        rescale_factor,
        resampling_method,
):
    if clicks is None:
        raise PreventUpdate

    display_cfg = cache_storage.cache['image']
    # Modify in memory
    display_cfg[DataDisplayCfgKey.RESCALE_FACTOR.value] = rescale_factor
    display_cfg[DataDisplayCfgKey.RESAMPLING_METHOD.value] = int(resampling_method)
    cache_storage.cache.set('image', display_cfg)

    print("ON Confirm")
    print(cache_storage.cache['image'])

    return dict(
        show_modal=False,
        ui_trigger=True,
    )
