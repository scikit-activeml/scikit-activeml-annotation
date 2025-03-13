from typing import Any
import base64
from io import BytesIO

import dash
from dash import html, dcc, Input, Output, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px

import numpy as np

from PIL import Image

from hydra.utils import instantiate

from util.deserialize import compose_config
from core.api import request_query, completed_batch, get_human_readable_sample
from core.schema import *
from core.adapter import *
from ui.storekey import StoreKey

dash.register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)


def encode_base64(image: np.ndarray) -> str:
    # Normalize the image to 0–255 range and convert to uint8
    if image.max() > 1:  # Handle images already in range 0-255
        image = 255 * (image / image.max())
    image = image.astype(np.uint8)

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(image)
    # print(pil_image.size)
    pil_image = pil_image.resize((250, 250), Image.Resampling.NEAREST)

    # Save image to a BytesIO stream in PNG format
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the image as Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the data URI
    return f"data:image/png;base64,{image_base64}"


# Triggers with each reload.
def load_image(bunch, idx: int) -> bytes:
    images = bunch.images

    image = images[idx]
    return image
    # image_url = encode_base64(images[idx])
    # return image_url


def load_text(bunch, idx: int):
    pass


# TODO No longer needed.
def get_label_names(bunch) -> list[str]:
    if 'target_names' in bunch:
        label_names = bunch.target_names

    # Some dataset dont return target_names
    label_names = np.unique(bunch.target)
    return label_names


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

    # Todo overrides of lower lvl config can be done like so:
    # cfg = compose(config_name="config", overrides=["database.host=remote_server"])

    selections = store_data[StoreKey.SELECTIONS.value]

    overrides = {
        'dataset': dataset_id,
        'query_strategy': selections['qs_id'],
        'model': selections['model_id']
    }

    activeMl_cfg = compose_config(overrides)
    dataset_cfg = activeMl_cfg.dataset
    # TODO data is loaded over and over again. when it not even needed.
    # adapter: DataLoaderAdapter = instantiate(dataset_cfg.adapter_cfg, _recursive_=False)
    # TODO make naming consistent preprocessor/ adapter wtf.
    adapter: BaseAdapter = instantiate(activeMl_cfg.preprocessor.definition)

    # TODO this will have to change if one file contains multiple samples.
    X, file_names = adapter.get_or_compute_embeddings(activeMl_cfg.dataset)

    # TODO Human Readable data
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
    # human_data_path = adapter.get_human_data(query_idx)
    # human_data_path = get_human_readable_sample(dataset_cfg, query_idx)
    human_data_path = file_names[query_idx]

    # bunch = instantiate(dataset_cfg.human_adapter)
    # label_names = get_label_names(bunch)
    # image = load_image(bunch, query_idx)

    # print("data after setup page is: ", data)
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

    # Override existing batchstate
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
