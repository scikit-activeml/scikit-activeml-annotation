from typing import Any

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
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.express as px

from hydra.utils import instantiate

from util.deserialize import compose_config
from core.api import (
    request_query,
    completed_batch,
    get_or_compute_embeddings
)
from core.schema import *
from core.adapter import *
from ui.storekey import StoreKey

register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)


def layout(**kwargs):
    return html.Div([
        dcc.Location(id='url-annotation', refresh=True),
        dmc.AppShell(
            [
                dmc.AppShellNavbar(
                    id="sidebar-container-annotation",
                    children=[
                        # Place your navigation items here
                        html.Div("Navigation Item 1"),
                        html.Div("Navigation Item 2"),
                    ],
                    p="md",
                    style={'border': '4px solid red'}
                ),
                dmc.AppShellMain(
                    id="hero-container-annotation",
                    children=dmc.Stack(
                        children=[
                            # Insert your main content components here
                            html.Div("Main Content")
                        ],
                    ),
                    style={'border': '4px solid blue'}
                ),
                dmc.AppShellAside(
                    id="right-panel",
                    children=[
                        # Place additional content or panels here
                        html.Div("Aside Content")
                    ],
                    p="md",
                    style={'border': '4px solid red'}
                ),
            ],
            navbar={
                "width": 200,
                "breakpoint": "sm",
                "collapsed": {"mobile": True},
            },
            aside={
                "width": 200,
                "breakpoint": "sm",
                "collapsed": {"mobile": True},
            },
            padding="md",
            id="appshell",
        )
    ])


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


# TODO make components out of these.
def display_image(path_to_img):
    # Use a separate Callback to update the image.
    image = Image.open(path_to_img).convert("RGB")

    # TODO why does this loading not work?
    return (
        dcc.Graph(
            figure=px.imshow(
                image,
                labels={},
                # color_continuous_scale='gray'
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


def create_chip(idx, label, probability=None):
    chip = dmc.Chip(
        label,
        value=str(idx),
        styles={"label": {"textAlign": "center"}},  # Ensures label is centered
    )
    if probability is None:
        return chip

    return dmc.InputWrapper(
        chip,
        inputWrapperOrder=['input', 'label', 'description'],
        # description=dmc.Text(f"{probability:.2f}", size="xs"),
        description=f"{probability:.2f}",
        style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
    )


def create_chip_group(label_names, class_prob):
    if class_prob is None:
        chips = [create_chip(idx, label) for idx, (label, probability) in enumerate(label_names)]
        preselect = '0'
    else:
        highest_prob_idx = np.argmax(class_prob)
        preselect = str(highest_prob_idx)
        chips = [create_chip(idx, label, probability) for idx, (label, probability) in
                 enumerate(zip(label_names, class_prob))]

    chip_group = dmc.ChipGroup(
        children=chips,
        multiple=False,
        value=preselect,  # Default value
        id="label-radio",
    )

    return dmc.ScrollArea(
        dmc.Box(
            dmc.Group(
                chip_group,
                justify="flex-start",
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "10px",  # space between chips
                },
            ),
            style={"display": "flex"}
        ),
        style={
            "width": "100%",
            "height": "200px",  # Adjust height as needed
        },
    )


# TODO seperate Loading Image from
def create_hero_section(label_names: list[str], dataset_cfg: DatasetConfig, human_data: Any, batch: Batch, progress: float):
    # TODO instantiate the data_type enum somewhere else
    data_type: DataType = instantiate(dataset_cfg.data_type)

    if data_type.value == DataType.IMAGE.value:
        rendered_data = display_image(human_data)
    elif data_type.value == DataType.TEXT.value:
        rendered_data = display_text(human_data)
    else:
        rendered_data = display_image(human_data)

    class_prob = None
    if batch.class_probas:
        class_prob = batch.class_probas[batch.progress]

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
                            dmc.ScrollArea(
                                create_chip_group(label_names, class_prob),
                                style={
                                    "width": "100%",
                                    "height": "100px",  # Adjust the height based on your needs
                                },
                                type='hover'
                            )
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
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        'Confirm Selection',
                                        id='confirm-button',
                                        color='dark',
                                    ),
                                    width="auto"
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        'Mark as Outlier',
                                        id='outlier-button',
                                        color='dark',
                                    ),
                                    width="auto"
                                ),
                            ],
                            justify="center"  # Center the buttons in the row
                        ),
                        style={'textAlign': 'center'},
                    ),
                    style={'marginBottom': '10px'},
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


@callback(
    Input('url-annotation', 'pathname'),
    State('session-store', 'data'),
    output=dict(
        session_store=Output('session-store', 'data', allow_duplicate=True),
        sidebar_container=Output('sidebar-container-annotation', 'children'),
        hero_container=Output('hero-container-annotation', 'children'),
    ),
    prevent_initial_call=True
)
def setup_annotations_page(pathname, store_data):
    dataset_id = pathname.split('/')[-1]
    print("[Annot] init annotation page with dataset: ", dataset_id)
    session_cfg = SessionConfig(batch_size=5)

    # info overrides of lower lvl config can be done like so:
    # cfg = compose(config_name="config", overrides=["database.host=remote_server"])

    overrides = {
        'dataset': store_data[StoreKey.DATASET_SELECTION.value],
        'query_strategy': store_data[StoreKey.QUERY_SELECTION.value],
        'adapter': store_data[StoreKey.ADAPTER_SELECTION.value],
        '+model': store_data[StoreKey.MODEL_SELECTION.value]  # add model to default list
    }

    print(overrides)
    # TODO clean this mess up.
    activeMl_cfg = compose_config(overrides)
    dataset_cfg = activeMl_cfg.dataset
    adapter_cfg = activeMl_cfg.adapter

    # TODO this will have to change if one file contains multiple samples.
    # TODO avoid reloading X and file_names every time. Only needed when.
    # This should only be needed when a batch is completed.
    X, file_names = get_or_compute_embeddings(dataset_cfg, adapter_cfg)
    print("Shape of X:", X.shape)

    if StoreKey.BATCH_STATE.value not in store_data:
        # New Session
        batch = request_query(activeMl_cfg, session_cfg, X, file_names)
        store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    else:
        # Existing Session
        batch: Batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
        batch_completed = len(batch.indices) <= batch.progress
        if batch_completed:
            print("BATCH IS COMPLETED")
            # Store labeling data to disk
            # TODO to much serialization deserialization
            completed_batch(dataset_id, batch)

            # Initialize the next batch
            batch = request_query(activeMl_cfg, session_cfg, X, file_names)
            store_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    print("Batch")
    print('indices:', batch.indices)
    print('progress', batch.progress)
    print('annotations:', batch.annotations)

    idx = batch.progress
    # query_idx -> file_name
    query_idx = batch.indices[idx]
    progress_percent = idx / len(batch.indices)

    # TODO generalize. How the human readable data and how the label names are fetched.
    # From Cache?
    label_names = dataset_cfg.label_names

    # TODO maybe the adapter should be responsible with specifying how to get human representation for sample with idx
    human_data_path = file_names[query_idx]

    return dict(
        session_store=store_data,
        sidebar_container=create_sidebar(),
        hero_container=create_hero_section(label_names, dataset_cfg, human_data_path, batch, progress_percent)
    )


@callback(
    Input('confirm-button', 'n_clicks'),
    Input('outlier-button', 'n_clicks'),
    State('label-radio', 'value'),
    State('session-store', 'data'),
    output=dict(
        session_data=Output('session-store', 'data', allow_duplicate=True),
        pathname=Output('url-annotation', 'pathname'),
    ),
    prevent_initial_call=True,
)
def on_button_click(
    confirm_clicks: int,
    outlier_clicks: int,
    value: int,
    session_data: dict
):
    # TODO is this needed?
    if (confirm_clicks is None or confirm_clicks == 0) and (outlier_clicks is None or outlier_clicks == 0):
        raise PreventUpdate

    trigger_id = callback_context.triggered_id
    if trigger_id == "confirm-button":
        annotation = int(value)
    else:
        annotation = np.inf

    print("Annotated label: ", annotation)

    # Update the Session's Batch State
    batch_state_json = session_data[StoreKey.BATCH_STATE.value]
    batch_state: Batch = Batch.from_json(batch_state_json)
    # print("pre", batch_state)
    idx = batch_state.progress
    batch_state.annotations[idx] = annotation
    batch_state.progress += 1

    # Override existing batch_state
    session_data[StoreKey.BATCH_STATE.value] = batch_state.to_json()

    return dict(
        session_data=session_data,
        pathname=None  # Refresh page by passing None.
    )


# TODO add cleanup function when switching away from annot page.


