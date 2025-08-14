from dash import (
    html,
    dcc,
    register_page,
    callback,
    Input,
    Output,
    State,
    callback_context
)

from core.api import *
from ui.common import compose_from_state
from util.deserialize import *

from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc

from ui.storekey import StoreKey

register_page(
    __name__,
    path_template='/embedding',
    description='Embedding Page',
)


def layout(**kwargs):
    return (
        dmc.AppShellMain(
            [
                dcc.Location('url-embedding', refresh=True),
                dcc.Location('url-embedding-init', refresh=False),
                dmc.Stack(
                    [
                        dmc.Title("Embedding", id='embedding-title'),

                        dmc.Container(
                            id="embedding-selection-container"
                        ),

                        dmc.Group(
                            [
                                dmc.Button("Cancel", id='cancel-embedding-button', disabled=True),
                                dmc.Button("Start Embedding", id='embedding-button')
                            ],
                            id='embedding-button-container'
                        ),

                        dmc.Progress(
                            id="embedding-progress",
                            value=0,
                            size="xl",
                            animated=True,
                            style={
                                'width': '75%',
                                'height': '30px'
                            },
                            transitionDuration=500
                        ),
                    ],
                    align='center'
                ),
            ]
        )
    )


@callback(
    Input('url-embedding-init', 'pathname'),
    State('session-store', 'data'),
    output=dict(
        embedding_selection_content=Output("embedding-selection-container", 'children')
    )
)
def setup_page(
        _,
        session_data
):
    print("setup embedding page callback")
    return dict(
        embedding_selection_content=create_selected_embedding_view(session_data)
    )


@callback(
    Input("embedding-button", 'n_clicks'),
    output=dict(
        title=Output('embedding-title', 'children'),
        # cancel_disabled=Output("cancel-embedding-button", 'disabled')
    ),
    prevent_initial_call=True,
)
def on_embedding_start(
        n_clicks,
):
    print("on_embedding_start callback")
    return dict(
        title="Embedding in progress...",
        # cancel_disabled=False
    )


@callback(
    Input('cancel-embedding-button', 'n_clicks'),
    output=dict(
        title=Output('embedding-title', 'children', allow_duplicate=True),
        progress=Output('embedding-progress', 'value'),
        # cancel_disabled=Output("cancel-embedding-button", 'disabled', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_cancel(
        n_clicks
):
    return dict(
        title="Embedding",
        progress=0,
        # cancel_disabled=True
    )


@callback(
    Input("embedding-button", 'n_clicks'),
    State("session-store", 'data'),
    progress=Output('embedding-progress', 'value'),
    cancel=Input('cancel-embedding-button', 'n_clicks'),
    running=[
        (Output('embedding-button', 'loading'), True, False),
        (Output('cancel-embedding-button', 'disabled'), False, True),
        (Output('embedding-progress', 'animated'), True, False)
    ],
    output=dict(
        title=Output('embedding-title', 'children', allow_duplicate=True),
        embedding_button_container=Output('embedding-button-container', 'children')
    ),
    background=True,
    prevent_initial_call=True,
)
def compute_embedding(
        progress_func,
        n_clicks,
        store_data,
):
    if n_clicks is None:
        raise PreventUpdate

    print("compute embedding background callback")
    _compute_embedding(store_data, progress_func)

    return dict(
        title="Embedding completed!",
        embedding_button_container=create_change_page_buttons()
    )


def _compute_embedding(store_data, progress_func):
    activeml_cfg = compose_from_state(store_data)
    compute_embeddings(activeml_cfg, progress_func)


def create_selected_embedding_view(session_data):
    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
    embedding_id = session_data[StoreKey.EMBEDDING_SELECTION.value]

    return (
        dmc.Card(
            [
                dmc.Text(f'Dataset: {dataset_id}'),
                dmc.Text(f'Embedding: {embedding_id}'),
            ],
        )
    )


def create_change_page_buttons():
    home_button = dmc.Button('Home', id='go-home-button')
    annot_button = dmc.Button("Annotation", id='go-annotating-button')
    return [home_button, annot_button]


@callback(
    Input('go-home-button', 'n_clicks'),
    Input('go-annotating-button', 'n_clicks'),
    State('session-store', 'data'),
    output=dict(
        pathname=Output('url-embedding', 'pathname')
    ),
    prevent_initial_call=True
)
def change_page(
        home_clicks,
        annot_clicks,
        session_data
):
    if home_clicks is None and annot_clicks is None:
        raise PreventUpdate

    print("Change page callback")
    trigger_id = callback_context.triggered_id

    if trigger_id == 'go-home-button':
        pathname = '/'
    else:
        # go-annotating-button
        dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
        pathname = f'/annotation/{dataset_id}'

    return dict(
        pathname=pathname
    )



