from typing import Any

import dash
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

from hydra.utils import instantiate

from skactiveml.utils import MISSING_LABEL

from util.deserialize import compose_config
from core.api import (
    request_query,
    completed_batch,
    get_embeddings
)
from core.schema import *
from ui.storekey import StoreKey
from ui.components.data_display import *

register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)

# TODO create variables for the id's


def layout(**kwargs):
    return html.Div([
        dcc.Location(id='url-annotation', refresh=True),
        dcc.Store(id='last-batch-store', storage_type='session'),
        dmc.AppShell(
            [
                dmc.AppShellNavbar(
                    id="sidebar-container-annotation",
                    children=[
                        create_sidebar()
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
                    style={'border': '5px dotted blue'}
                ),
                dmc.AppShellAside(
                    id="right-panel",
                    children=[
                        dmc.Stack(
                            [
                                dmc.SemiCircleProgress(
                                    label="Annotation Progress",
                                    id='annotation-progress-circle',
                                    value=40,
                                )
                            ],
                            align='center'
                        )
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
        dmc.Stack(
            [
                dmc.Center(
                    dmc.Text("Settings"),
                ),

                # Batch Size selection
                dmc.NumberInput(
                    label="Batch Size",
                    id='batch-size-input',
                    allowNegative=False,
                    debounce=True,
                    value=5,
                    required=True,
                    persistence='batch-size-persistence',
                    persistence_type='local',
                ),

                # Subsampling selection
                dmc.Text("Subsampling"),
                dmc.Flex(
                    [
                        dmc.NumberInput(
                            # label="Subsampling",
                            id='subsampling-input',
                            allowNegative=False,
                            debounce=True,
                            hideControls=True
                        ),
                        dmc.Switch(
                            id='my-checkbox',
                            checked=False
                        )
                    ],
                    direction="row",
                    align="center",  # centers items vertically
                    gap="md"         # optional, for spacing between elements
                ),

                dmc.Text(
                    'Query Strategy'
                ),

                # Skip Button
                dmc.Center(
                    dmc.Button(
                        "Skip Batch",
                        id="skip-batch-button",
                    ),
                ),
            ],
            style={'border': '2px solid red'},
        )
    )


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


def create_chip_group(classes, batch, class_prob):
    # Check if there is some annotation already for that sample in case the user used back btn.
    annotation = batch.annotations[batch.progress]
    was_annotated = annotation is not None
    was_labaled = was_annotated and isinstance(annotation, int)

    if class_prob is None:
        chips = [create_chip(idx, label) for idx, label in enumerate(classes)]
        preselect = str(annotation) if was_labaled else None
    else:
        if was_labaled:
            preselect = str(annotation)
        elif was_annotated:
            preselect = None
        else:
            highest_prob_idx = np.argmax(class_prob)
            preselect = str(highest_prob_idx)

        chips = [create_chip(idx, label, probability) for idx, (label, probability) in
                 enumerate(zip(classes, class_prob))]

    chip_group = dmc.ChipGroup(
        children=chips,
        multiple=False,
        value=preselect,
        id="label-radio",
    )

    return dmc.ScrollArea(
        dmc.Box(
            dmc.Flex(
                chip_group,
                wrap='wrap',
                justify='flex-start',
                gap='10px'
            ),

            style={
                'maxHeight': '40vh',
            }
        ),

        type='auto',
        offsetScrollbars=True,
        style={
            "width": "100%",
            'border': 'green dashed 3px',
        },
    )


# TODO seperate Loading Image from
# TODO rename this function to have a better name.
def create_hero_section(classes: list[str], dataset_cfg: DatasetConfig, human_data_path: str, batch: Batch, progress: float):
    # TODO instantiate the data_type enum somewhere else
    data_type: DataType = instantiate(dataset_cfg.data_type)

    human_data_path = ROOT_PATH / human_data_path

    if data_type.value == DataType.IMAGE.value:
        rendered_data = create_image_display(human_data_path)
    elif data_type.value == DataType.TEXT.value:
        rendered_data = create_text_display(human_data_path)
    else:
        rendered_data = create_audio_display(human_data_path)

    class_prob = None
    if batch.class_probas:
        class_prob = batch.class_probas[batch.progress]

    return (
        dmc.Container(
            [
                # Data display Container
                dmc.Stack(
                    # dmc.Container(
                        rendered_data,
                        style={'border': '4px dotted pink'},
                    # ),
                    align="center",
                ),

                # Label selection
                dmc.Group(
                    dmc.Stack(
                        [
                            dmc.Title('Select Label', order=4),
                            create_chip_group(classes, batch, class_prob),
                        ],
                        style={
                            'textAlign': 'center',
                            # 'marginTop': '5px'
                        },
                        gap='xs'
                    ),
                    # style={
                    #     'border': 'red solid 4px'
                    # }
                ),

                # Confirm button
                dmc.Group(
                    [
                        dmc.Button(
                            'Back',
                            id="back-button",
                            color='dark'
                        ),

                        dmc.Button(
                            'Discard',
                            id='discard-button',
                            color='dark'
                        ),

                        dmc.Button(
                            'Skip',
                            id="skip-button",
                            color='dark'
                        ),

                        dmc.Button(
                            'Confirm',
                            id='confirm-button',
                            color='dark'
                        ),

                    ],
                    style={'border': 'red dashed 2px'},
                    justify='center',
                ),

                # Progress bar
                html.Div(
                    [
                        # The Mantine Progress bar with dynamic section
                        dmc.ProgressRoot(
                            dmc.ProgressSection(
                                value=progress * 100,
                                color="blue",
                                # animated=True,
                                # striped=True
                            ),
                            radius=25,
                            size="lg",
                            style={"height": "40px"},
                        ),
                        # The overlay text: always centered
                        html.Div(
                            "Batch Progress",
                            style={
                                "position": "absolute",
                                "width": "100%",
                                "top": "50%",
                                "left": "50%",
                                "transform": "translate(-50%, -50%)",
                                "textAlign": "center",
                                "color": "white",
                                "pointerEvents": "none",
                            },
                        ),
                    ],
                    style={"position": "relative", "width": "100%"},
                ),
            ],
            fluid=True,
            style={'border': '4px dashed green'},
        )
    )


@callback(
    Input('url-annotation', 'pathname'),
    State('session-store', 'data'),
    State('batch-size-input', 'value'),
    State('subsampling-input', 'value'),
    output=dict(
        session_store=Output('session-store', 'data', allow_duplicate=True),
        hero_container=Output('hero-container-annotation', 'children'),
        last_batch=Output('last-batch-store', 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def setup_annotations_page(
        pathname,
        store_data,
        batch_size,
        subsampling,
):
    dataset_id = pathname.split('/')[-1]
    print("[Annot] init annotation page with dataset: ", dataset_id)

    session_cfg = SessionConfig(batch_size=batch_size, subsampling=subsampling)

    # info overrides of lower lvl config can be done like so:
    # cfg = compose(config_name="config", overrides=["database.host=remote_server"])

    overrides = {
        'dataset': store_data[StoreKey.DATASET_SELECTION.value],
        'query_strategy': store_data[StoreKey.QUERY_SELECTION.value],
        'adapter': store_data[StoreKey.ADAPTER_SELECTION.value],
        '+model': store_data[StoreKey.MODEL_SELECTION.value]  # add model to default list
    }
    print(overrides)

    # TODO avoid reloading X and file_names every time. Cache could make sense.
    activeMl_cfg = compose_config(overrides)
    dataset_cfg = activeMl_cfg.dataset

    # TODO This should only be needed when a batch is completed.
    X, file_names = get_embeddings(activeMl_cfg)
    last_batch_json = dash.no_update

    if StoreKey.BATCH_STATE.value not in store_data:
        # New Session
        batch = request_query(activeMl_cfg, session_cfg, X, file_names)
        store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    else:
        # Existing Session
        batch: Batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
        if batch.is_completed():
            print("BATCH IS COMPLETED")
            # Store labeling data to disk
            # TODO to much serialization deserialization

            completed_batch(dataset_id, batch)

            # Override last batch
            last_batch_json = batch.to_json()
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
    classes = dataset_cfg.classes

    # TODO maybe the adapter should be responsible with specifying how to get human representation for sample with idx
    human_data_path = file_names[query_idx]
    print('human data path')
    print(human_data_path)

    return dict(
        session_store=store_data,
        hero_container=create_hero_section(classes, dataset_cfg, human_data_path, batch, progress_percent),
        last_batch=last_batch_json
    )


@callback(
    Input('confirm-button', 'n_clicks'),
    Input('discard-button', 'n_clicks'),
    Input('skip-button', 'n_clicks'),
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
    skip_clicks: int,
    value: int,
    session_data: dict
):
    # TODO is this needed?
    if ((confirm_clicks is None or confirm_clicks == 0) and
        (outlier_clicks is None or outlier_clicks == 0) and
            (skip_clicks is None or skip_clicks == 0)):
        raise PreventUpdate

    trigger_id = callback_context.triggered_id
    if trigger_id == "confirm-button":
        annotation = int(value)
    elif trigger_id == "skip-button":
        annotation = MISSING_LABEL
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


@callback(
    Input('skip-batch-button', 'n_clicks'),
    State('session-store', 'data'),
    output=dict(
        session_data=Output('session-store', 'data', allow_duplicate=True),
        pathname=Output('url-annotation', 'pathname', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_skip_batch(
    n_clicks: int,
    session_data: dict,
):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    # reset batch state
    batch_json = session_data.pop(StoreKey.BATCH_STATE.value, None)
    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]

    # Store annotations that have been made so far.
    batch = Batch.from_json(batch_json)

    for idx, val in enumerate(batch.annotations):
        if val is None:
            batch.annotations[idx] = MISSING_LABEL

    completed_batch(dataset_id, batch)

    return dict(
        session_data=session_data,
        pathname=None  # By passing None the page is reloaded
    )


@callback(
    Input('back-button', 'n_clicks'),
    State('session-store', 'data'),
    State('last-batch-store', 'data'),
    output=dict(
        session_data=Output('session-store', 'data'),
        pathname=Output('url-annotation', 'pathname', allow_duplicate=True),
        last_batch=Output('last-batch-store', 'data', allow_duplicate=True),
    ),
    prevent_initial_call=True
)
def on_back_clicked(
    clicks,
    session_data,
    last_batch,
):
    if clicks is None:
        raise PreventUpdate

    print("on back click callback")
    batch = Batch.from_json(session_data[StoreKey.BATCH_STATE.value])

    if batch.progress == 0:
        print("Have to get last batch to be able to go back.")
        if last_batch is None:
            print("CANNOT go back further!")
            raise PreventUpdate

        batch = Batch.from_json(last_batch)
        last_batch = None
    else:
        last_batch = dash.no_update

    batch.progress -= 1
    session_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    return dict(
        session_data=session_data,
        pathname=None,  # None to refresh the page.
        last_batch=last_batch
    )


# TODO add cleanup function when switching away from annot page.


