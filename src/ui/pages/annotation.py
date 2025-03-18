from typing import Any

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
from hydra.utils import instantiate

from util.deserialize import compose_config
from core.api import request_query, completed_batch
from core.schema import *
from core.adapter import *
from ui.storekey import StoreKey

dash.register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)


def layout(**kwargs):
    return (
        # dcc.Loading(
        html.Div(
            [
                dcc.Location(id='url-annotation', refresh=True),
                dbc.Row(
                    [
                        dbc.Col(
                            id="sidebar-container-annotation",
                            width=3,
                            style={'border': '4px solid red'}
                        ),
                        dbc.Col(
                            id="hero-container-annotation",
                            width=6,
                            class_name='d-flex justify-content-start',
                            style={'border': '4px solid blue'}
                        ),
                        dbc.Col(
                            width=3,
                            style={'border': '4px solid red'}
                        )
                    ],
                    # style={'border': '2px solid red'}
                    class_name='px-0',
                    justify='start'
                )
            ],
            # style={'border': '2px dashed black'},
        )
        # ),
    )


def create_sidebar():
    return (
        dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Settings"),
                        dbc.CardBody([
                            # Batch Size selection
                            html.Label('Batch Size'),
                            dcc.Input(
                                id='batch-size-input',
                                type='number',
                                value=10,
                                min=1,
                                step=1,
                                style={'width': '100%'}
                            ),
                            html.Br(),
                            html.Br(),
                            # Subsampling selection
                            html.Label('Subsampling'),
                            dcc.Input(
                                id='subsampling-input',
                                type='number',
                                value=1000,
                                min=1,
                                step=1,
                                style={'width': '100%'}
                            ),
                            html.Br(),
                        ])
                    ],
                )
            ],
            style={'border': '2px solid red'},
            # width=3
        )
    )


def display_image(path_to_img):
    image = Image.open(path_to_img).convert("RGB")

    return (
        dcc.Loading(
            dcc.Graph(
                figure=px.imshow(
                    image,
                    labels={},
                    # color_continuous_scale='gray'
                ),
            ),
        ),
    )


def display_text(text):
    return (
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    dcc.Markdown(
                        text,  # Provide your text data here
                        className="markdown-content",
                        # Add additional Markdown options if necessary
                    ),
                    className="p-3",
                ),
                className="justify-content-center",
            ),
            className="text-container",
        )
    )


def display_audio(audio):
    print(audio)
    raise NotImplementedError


def create_hero_section(label_names: list[str], dataset_cfg: DatasetConfig, human_data: Any, progress: float):
    # TODO instantiate the data_type enum somewhere else
    data_type: DataType = instantiate(dataset_cfg.data_type)

    if data_type.value == DataType.IMAGE.value:
        rendered_data = display_image(human_data)
    elif data_type.value == DataType.TEXT.value:
        rendered_data = display_text(human_data)
    else:
        rendered_data = display_image(human_data)

    return (
        dbc.Container(
            [
                # Data display
                dbc.Row(
                    dbc.Col(
                        rendered_data
                    ),
                    # style={'marginBottom': '5px'},
                    style={'border': '4px dotted pink'}
                ),

                # Label selection
                dbc.Row(
                    dbc.Col(
                        [
                            html.H4('Select Label'),
                            dcc.RadioItems(
                                id='label-radio',
                                options=[{'label': l_name, 'value': idx} for idx, l_name in enumerate(label_names)],
                                value=0,  # Default to the first label
                                labelStyle={
                                    'display': 'block',
                                    'margin': '10px 0'
                                },
                            ),
                        ],
                        style={
                            'textAlign': 'center',
                            # 'marginTop': '5px'
                        },
                    ),
                    style={'marginBottom': '20px'},
                ),

                # Confirm button
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            'Confirm Selection',
                            id='confirm-button',
                            color='dark',
                        ),
                        style={'textAlign': 'center'},
                    ),
                    style={'marginBottom': '20px'},
                ),

                # Progress bar
                dbc.Row(
                    dbc.Col(
                        dbc.Progress(
                            id='batch-progress-annotation',
                            label="Batch Progress",
                            min=0,
                            max=1,
                            value=progress,
                        ),
                    ),
                    style={'marginBottom': '10px'}
                ),
            ],
            fluid=True,
            style={'border': '4px dashed green'},
        )
    )


@dash.callback(
    Output('session-store', 'data', allow_duplicate=True),
    Output('sidebar-container-annotation', 'children'),
    Output('hero-container-annotation', 'children'),
    Input('url-annotation', 'pathname'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def setup_annotations_page(pathname, store_data):
    dataset_id = pathname.split('/')[-1]
    print("[Annot] init annotation page with dataset: ", dataset_id)
    session_cfg = SessionConfig(batch_size=5)

    # info overrides of lower lvl config can be done like so:
    # cfg = compose(config_name="config", overrides=["database.host=remote_server"])

    selections = store_data[StoreKey.SELECTIONS.value]

    overrides = {
        'dataset': dataset_id,
        'query_strategy': selections['qs_id'],
        'model': selections['model_id']
    }

    activeMl_cfg = compose_config(overrides)
    dataset_cfg = activeMl_cfg.dataset
    adapter: BaseAdapter = instantiate(activeMl_cfg.adapter.definition)
    print("Selected adapter:", type(adapter))

    # TODO this will have to change if one file contains multiple samples.
    X, file_names = adapter.get_or_compute_embeddings(activeMl_cfg.dataset)
    print("Shape of X:", X.shape)

    if StoreKey.BATCH_STATE.value not in store_data:
        # New Session
        batch = request_batch(activeMl_cfg, session_cfg, X, file_names)
        # store_data = {StoreKey.BATCH_STATE.value: batch.to_json()}
        store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    else:
        # Existing Session
        batch: Batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
        batch_completed = len(batch.indices) <= batch.progress
        if batch_completed:
            print("BATCH IS COMPLETED")
            # Store labeling data to disk
            completed_batch(dataset_id, batch)

            # Initialize the next batch
            batch = request_batch(activeMl_cfg, session_cfg, X, file_names)
            store_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    idx = batch.progress
    # query_idx -> file_name
    query_idx = batch.indices[idx]
    progress = idx / len(batch.indices)

    # TODO generalize. How the human readable data and how the label names are fetched.
    # From Cache?
    label_names = dataset_cfg.label_names

    # TODO maybe the adapter should be responsible with specifying how to get human representation for sample with idx
    human_data_path = file_names[query_idx]
    return store_data, create_sidebar(), create_hero_section(label_names, dataset_cfg, human_data_path, progress)


@dash.callback(
    Output('session-store', 'data', allow_duplicate=True),
    Output('url-annotation', 'pathname'),
    Input('confirm-button', 'n_clicks'),
    State('label-radio', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True,
)
def on_button_click(n_clicks: int, value: int, session_data: dict):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    print("Annotated label: ", value)

    # Update the Session's Batch State
    batch_state_json = session_data[StoreKey.BATCH_STATE.value]
    batch_state: Batch = Batch.from_json(batch_state_json)
    # print("pre", batch_state)
    idx = batch_state.progress
    batch_state.annotations[idx] = value
    batch_state.progress += 1
    print("post", batch_state)

    # Override existing batch_state
    session_data[StoreKey.BATCH_STATE.value] = batch_state.to_json()

    # Refresh page by passing None.
    return session_data, None


# Helper
def request_batch(
    cfg: ActiveMlConfig,
    session_cfg: SessionConfig,
    X: np.ndarray,
    file_names: list[str]
) -> Batch:
    query_indices = request_query(cfg, session_cfg, X, file_names)
    batch_state = Batch(
        indices=query_indices.tolist(),
        progress=0,
        annotations=[None] * len(query_indices)
    )
    return batch_state
