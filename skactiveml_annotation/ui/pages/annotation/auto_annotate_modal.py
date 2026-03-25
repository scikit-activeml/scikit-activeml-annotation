from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
)

from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from skactiveml_annotation import util
from skactiveml_annotation.core import api
from skactiveml_annotation.shared_ids import (
    SELECTION,
)

from skactiveml_annotation.ui import common
from skactiveml_annotation.ui.pages.home.selection import Selection


from . import ids

AUTO_ANNOTATE_MODAL = { 'type': 'modal', 'index': "AutoAnnotateModal" }


def create_auto_annotate_modal():
    return dmc.Modal(
        dmc.Stack(
            [

                dmc.NumberInput(
                    id=ids.AUTO_ANNOTATE_THRESHOLD,
                    min=0,
                    max=1,
                    hideControls=True,
                    label="Threshold",
                    placeholder="Enter a threshold",
                    value=0.99,
                    allowNegative=False,
                    w='35%',
                    required=True,
                    persistence=ids.AUTO_ANNOTATE_THRESHOLD,
                    persistence_type='session',
                ),

                dmc.Center(
                    dmc.Button(
                        'Confirm',
                        id=ids.AUTO_ANNOTATE_CONFIRM_BTN,
                        color='dark',
                    )
                )
            ],
        ),
        id=AUTO_ANNOTATE_MODAL,
        title='Auto Annotate with Threshold',
        centered=True,
        shadow='xl',
        withCloseButton=False,
    )


def register_callbacks(app: Dash):
    @app.callback(
        Input(ids.AUTO_ANNOTATE_BTN, 'n_clicks'),
        output=dict(
            modal_open=Output(AUTO_ANNOTATE_MODAL, 'opened', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def open_modal(
        clicks
    ):
        if clicks is None:
            raise PreventUpdate

        return dict(
            modal_open=True
        )
    _ = open_modal


    @callback(
        Input(ids.AUTO_ANNOTATE_CONFIRM_BTN, 'n_clicks'),
        State(SELECTION, 'data'),
        State(ids.AUTO_ANNOTATE_THRESHOLD, 'value'),
        output=dict(
            auto_annot_modal_open=Output(AUTO_ANNOTATE_MODAL, 'opened', allow_duplicate=True),
        ),
        prevent_initial_call=True,
        background=True,
    )
    def on_auto_annotate(
        click: int | None,
        selection_json: str,
        threshold: float,
    ):
        if click is None:
            raise PreventUpdate

        util.logging.setup_logging_background_callback()

        selection = Selection.model_validate_json(selection_json)
        activeml_cfg = common.compose_from_state(selection)
        X = api.load_embeddings(
            activeml_cfg.dataset.id,
            activeml_cfg.embedding.id
        )

        api.auto_annotate(X, activeml_cfg, threshold)

        return dict(
            auto_annot_modal_open=False,
        )
    _ = on_auto_annotate
