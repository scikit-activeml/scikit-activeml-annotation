
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
)
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from skactiveml_annotation import ui
from skactiveml_annotation.core import api
from skactiveml_annotation.ui.pages.home.selection import Selection

from skactiveml_annotation.core.shared_types import DashProgressFunc
from skactiveml_annotation.shared_ids import SELECTION

from skactiveml_annotation import util

from . import (
    ids
)


def register(app: Dash):
    @app.callback(
        Input(ids.INIT, 'pathname'),
        State(SELECTION, 'data'),
        output=dict(
            selection_container=Output(ids.SELECTION_CONTAINER, 'children')
        )
    )
    def setup_page(
        _,
        selection: Selection,
    ):
        return dict(
            selection_container=_create_selected_embedding_view(selection)
        )
    _ = setup_page


    @app.callback(
        Input(ids.CONFIRM_BUTTON, 'n_clicks'),
        output=dict(
            title=Output(ids.TITLE, 'children'),
            # cancel_disabled=Output(ids.CANCEL_BUTTON, 'disabled')
        ),
        prevent_initial_call=True,
    )
    def on_embedding_start(
        _,
    ):
        return dict(
            title="Embedding in progress...",
            # cancel_disabled=False
        )
    _ = on_embedding_start


    @app.callback(
        Input(ids.CANCEL_BUTTON, 'n_clicks'),
        output=dict(
            title=Output(ids.TITLE, 'children', allow_duplicate=True),
            progress=Output(ids.EMBEDDING_PROGRESS, 'value'),
            # cancel_disabled=Output(ids.CANCEL_BUTTON, 'disabled', allow_duplicate=True)
        ),
        prevent_initial_call=True
    )
    def on_cancel(
        _,
    ):
        return dict(
            title="Embedding",
            progress=0,
            # cancel_disabled=True
        )
    _ = on_cancel


    @callback(
        Input(ids.CONFIRM_BUTTON, 'n_clicks'),
        State(SELECTION, 'data'),
        progress=Output(ids.EMBEDDING_PROGRESS, 'value'),
        cancel=Input(ids.CANCEL_BUTTON, 'n_clicks'),
        running=[
            (Output(ids.CONFIRM_BUTTON, 'loading'), True, False),
            (Output(ids.CANCEL_BUTTON, 'disabled'), False, True),
            (Output(ids.EMBEDDING_PROGRESS, 'animated'), True, False)
        ],
        output=dict(
            title=Output(ids.TITLE, 'children', allow_duplicate=True),
            embedding_button_container=Output(ids.EMBEDDING_BTN_CONTAINER, 'children')
        ),
        background=True,
        prevent_initial_call=True,
    )
    def compute_embedding(
        progress_func: DashProgressFunc, # Progress func gets passed as first arg
        n_clicks: int | None,
        selection_json: str,
    ):
        if n_clicks is None:
            raise PreventUpdate

        # The background Callback runs in a different process.
        # Logging needs to be initialized in this new context
        util.logging.setup_logging_background_callback()

        selection = Selection.model_validate_json(selection_json)
        _compute_embedding(selection, progress_func)

        return dict(
            title="Embedding completed!",
            embedding_button_container=_create_change_page_buttons()
        )
    _ = compute_embedding


    @app.callback(
        Input(ids.GO_HOME_BUTTON, 'n_clicks'),
        Input(ids.GO_ANNOTATION_BUTTON, 'n_clicks'),
        State(SELECTION, 'data'),
        output=dict(
            pathname=Output(ids.URL, 'pathname')
        ),
        prevent_initial_call=True
    )
    def change_page(
        home_clicks: int | None,
        annot_clicks: int | None,
        selection: Selection,
    ):
        if home_clicks is None and annot_clicks is None:
            raise PreventUpdate

        trigger_id = callback_context.triggered_id

        if trigger_id == ids.GO_HOME_BUTTON:
            pathname = '/'
        elif trigger_id == ids.GO_ANNOTATION_BUTTON:
            pathname = f'/annotation/{selection.dataset_id}'
        else:
            raise RuntimeError(f"Unknown trigger_id {trigger_id}")

        return dict(
            pathname=pathname,
        )
    _ = change_page


def _compute_embedding(selection: Selection, progress_func: DashProgressFunc):
    activeml_cfg = ui.common.compose_from_state(selection)
    api.compute_embeddings(activeml_cfg, progress_func)


def _create_selected_embedding_view(selection: Selection):
    return (
        dmc.Card(
            [
                dmc.Text(f'Dataset: {selection.dataset_id}'),
                dmc.Text(f'Embedding: {selection.embedding_id}'),
            ],
        )
    )


def _create_change_page_buttons():
    home_button = dmc.Button('Home', id=ids.GO_HOME_BUTTON)
    annot_button = dmc.Button("Annotation", id=ids.GO_ANNOTATION_BUTTON)
    return [home_button, annot_button]
