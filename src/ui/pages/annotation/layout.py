import logging

import dash
import dash_mantine_components as dmc
from dash import (
    html,
    register_page,
    callback,
    Input,
    Output,
    State,
    callback_context,
    clientside_callback,
    ClientsideFunction,
    set_props,
    ALL
)
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from hydra.utils import instantiate

from skactiveml.utils import MISSING_LABEL

from ui import util
from util.deserialize import compose_config
from core.api import (
    request_query,
    completed_batch,
    load_embeddings,
    load_file_paths,
    get_total_num_samples,
    get_num_annotated,

)
from core.api import undo_annots_and_restore_batch

from core.schema import *
from ui.storekey import StoreKey, AnnotProgress

from ui.pages.annotation.ids import *
from ui.pages.annotation.components import *
from ui.pages.annotation.data_display import *

register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)


def compose_from_state(store_data) -> ActiveMlConfig:
    overrides = (
        ('dataset', store_data[StoreKey.DATASET_SELECTION.value]),
        ('query_strategy', store_data[StoreKey.QUERY_SELECTION.value]),
        ('embedding', store_data[StoreKey.EMBEDDING_SELECTION.value]),
        ('+model', store_data[StoreKey.MODEL_SELECTION.value])  # add model to default list
    )

    return compose_config(overrides)


def advance_batch(batch: Batch, annotation):
    idx = batch.progress
    batch.annotations[idx] = annotation
    batch.progress += 1


def layout(**kwargs):
    return (
        dmc.Box(
            [
                dcc.Location(id='url-annotation', refresh=True),
                dcc.Location(id=ANNOTATION_INIT, refresh=False),
                dcc.Store(id=UI_TRIGGER),
                dcc.Store(id=QUERY_TRIGGER),
                dcc.Store(id=ANNOT_PROGRESS),
                dmc.Box(id='label-radio'),  # avoid id error
                dmc.AppShell(
                    [
                        dmc.AppShellNavbar(
                            id="sidebar-container-annotation",
                            children=create_sidebar(),
                            p="md",
                            style={'border': '4px solid red'}
                        ),
                        dmc.Box(
                            children=[
                                dcc.Loading(
                                    children=[
                                        dmc.Box(
                                            id=DATA_DISPLAY_CONTAINER
                                        ),
                                    ],
                                    delay_hide=10,
                                    custom_spinner=dmc.Skeleton(
                                        visible=True,
                                        h="100%"
                                    )
                                ),

                                dmc.Group(
                                    [
                                        dmc.Title('Select Label', order=4),
                                        dmc.Tooltip(
                                            dmc.ActionIcon(
                                                DashIconify(icon="clarity:settings-line", width=20),
                                                variant="filled",
                                                id="label-setting-popup",
                                            ),
                                            label='Label settings',
                                        ),
                                        dmc.TextInput(placeholder='Search', id='label-search-text-input'),
                                    ],
                                    justify='center'
                                ),

                                dmc.Stack(
                                    id=LABELS_CONTAINER,
                                    # h='400px'
                                    align='center'
                                ),
                                create_confirm_buttons(),
                                create_progress_bar()
                            ],
                            style={
                                'border': '5px dotted blue',
                                 'height': '100%'
                            },
                            py=0,
                            px=150
                        ),
                        dmc.AppShellAside(
                            children=[
                                dmc.Stack(
                                    [
                                        dmc.Card(
                                            dmc.Stack(
                                                [
                                                    dmc.Group(
                                                        [
                                                            dmc.Text("Annotated:", style={"fontSize": "1vw"}),
                                                            dmc.Text(
                                                                dmc.NumberFormatter(
                                                                    id=ANNOT_PROGRESS_TEXT,
                                                                    thousandSeparator=' ',
                                                                ),
                                                                style={"fontSize": "1vw"}
                                                            ),
                                                        ],
                                                        gap=4
                                                    ),

                                                    dmc.Group(
                                                        [
                                                            dmc.Text("Total:", style={"fontSize": "1vw"}),
                                                            dmc.Text(
                                                                dmc.NumberFormatter(
                                                                    id=NUM_SAMPLES_TEXT,
                                                                    thousandSeparator=' '
                                                                ),
                                                                style={"fontSize": "1vw"}
                                                            )
                                                        ],
                                                        gap=4
                                                    )
                                                ],
                                                gap=5
                                            )
                                        )
                                    ],
                                    # p='xs',
                                    style={'border': '3px dashed green'},
                                    align='center'
                                )
                            ],
                            p="xs",
                            style={'border': '4px solid red'}
                        ),
                    ],
                    navbar={
                        "width": '13vw',
                        "breakpoint": "sm",
                        "collapsed": {"mobile": True},
                    },
                    aside={
                        "width": '13vw',
                        "breakpoint": "sm",
                        "collapsed": {"mobile": True},
                    },
                    padding=0,
                    id="appshell",
                )
            ],
            style={
                # 'height': '100%',
                'border': 'green dotted 5px'
            }
        )
    )


@callback(
    Input(ANNOTATION_INIT, 'pathname'),
    State('session-store', 'data'),
    output=dict(
        annot_progress=Output(ANNOT_PROGRESS, 'data'),
        ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True),
        query_trigger=Output(QUERY_TRIGGER, 'data', allow_duplicate=True),
    ),
    prevent_initial_call='initial_duplicate'
)
def init(
    _,
    store_data,
):
    batch_json = store_data.get(StoreKey.BATCH_STATE.value)
    if batch_json is None:
        return dict(
            ui_trigger=dash.no_update,
            query_trigger=True,
            annot_progress=init_annot_progress(store_data)
        )

    batch = Batch.from_json(batch_json)

    if batch.is_completed():
        return dict(
            ui_trigger=dash.no_update,
            query_trigger=True,
            annot_progress=init_annot_progress(store_data)
        )
    else:
        return dict(
            ui_trigger=True,
            query_trigger=dash.no_update,
            annot_progress=init_annot_progress(store_data)
        )


def init_annot_progress(store_data):
    dataset_id = store_data.get(StoreKey.DATASET_SELECTION.value)
    embedding_id = store_data.get(StoreKey.EMBEDDING_SELECTION.value)

    return {
        AnnotProgress.PROGRESS.value: get_num_annotated(dataset_id),
        AnnotProgress.TOTAL_NUM.value: get_total_num_samples(dataset_id, embedding_id)
    }


@callback(
    Input('confirm-button', 'n_clicks'),
    Input('discard-button', 'n_clicks'),
    Input('skip-button', 'n_clicks'),
    State('session-store', 'data'),
    State('label-radio', 'value'),
    State(ANNOT_PROGRESS, 'data'),
    output=dict(
        store_data=Output('session-store', 'data', allow_duplicate=True),
        annot_data=Output(ANNOT_PROGRESS, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_confirm(
    confirm_click,  # TODO use patter matching instead.
    discard_click,
    skip_click,
    store_data,
    value,
    annot_data
):
    if confirm_click is None and discard_click is None and skip_click is None:
        raise PreventUpdate

    trigger_id = callback_context.triggered_id
    if trigger_id == "confirm-button":
        annotation = int(value)
    elif trigger_id == "skip-button":
        annotation = MISSING_LABEL
    else:
        annotation = np.inf  # discarded

    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    advance_batch(batch, annotation)
    # Override existing batch
    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    if batch.is_completed():
        dataset_id = store_data[StoreKey.DATASET_SELECTION.value]
        embedding_id = store_data[StoreKey.EMBEDDING_SELECTION.value]

        num_annotated = completed_batch(dataset_id, batch, embedding_id)
        if num_annotated == annot_data[AnnotProgress.TOTAL_NUM.value]:
            print("ANNOTATION COMPLETE")
            raise PreventUpdate

        annot_data[AnnotProgress.PROGRESS.value] = num_annotated

        set_props(QUERY_TRIGGER, dict(data=True))
    else:
        set_props(UI_TRIGGER, dict(data=True))
        # TODO this wont work always if the user went back or skipped?
        # annot_data[AnnotProgress.PROGRESS.value] += 1

    return dict(
        store_data=store_data,
        annot_data=annot_data
    )


@callback(
    Input(UI_TRIGGER, 'data'),
    State('session-store', 'data'),
    output=dict(
        label_container=Output(LABELS_CONTAINER, 'children'),
        show_container=Output(DATA_DISPLAY_CONTAINER, 'children'),
        batch_progress=Output('batch-progress-bar', 'value'),
    ),
    prevent_initial_call=True,
)
def on_ui_update(
    ui_trigger,
    store_data
):
    if ui_trigger is None:
        raise PreventUpdate

    activeml_cfg = compose_from_state(store_data)
    file_paths = load_file_paths(activeml_cfg.dataset.id, activeml_cfg.embedding.id)
    data_type: DataType = instantiate(activeml_cfg.dataset.data_type)

    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    idx = batch.progress
    query_idx = batch.indices[idx]
    human_data_path = file_paths[query_idx]

    return dict(
        label_container=create_chip_group(activeml_cfg.dataset.classes, batch),
        show_container=create_data_display(data_type, human_data_path),
        batch_progress=(idx / len(batch.indices)) * 100,
    )


@callback(
    Input(QUERY_TRIGGER, 'data'),
    State('session-store', 'data'),
    State('batch-size-input', 'value'),
    State('subsampling-input', 'value'),
    output=dict(
        store_data=Output('session-store', 'data', allow_duplicate=True),
        ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True,
    # background=True,
    # TODO why is this loading so delayed?
    running=[
        (Output('confirm-button', 'loading'), True, False),
        (Output('discard-button', 'loading'), True, False),
        (Output('skip-button', 'loading'), True, False),
        (Output('back-button', 'loading'), True, False),
    ],
)
def on_query(
    trigger,
    store_data,
    batch_size,
    subsampling,
):
    if trigger is None:
        raise PreventUpdate

    print("on query")
    activeml_cfg = compose_from_state(store_data)
    X = load_embeddings(activeml_cfg.dataset.id, activeml_cfg.embedding.id)
    session_cfg = SessionConfig(batch_size, subsampling)

    batch = request_query(activeml_cfg, session_cfg, X)
    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    return dict(
        store_data=store_data,
        ui_trigger=True
    )


@callback(
    Input('skip-batch-button', 'n_clicks'),
    State('session-store', 'data'),
    output=dict(
        query_trigger=Output(QUERY_TRIGGER, 'data'),
        session_data=Output('session-store', 'data', allow_duplicate=True),
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
    embedding_id = session_data[StoreKey.EMBEDDING_SELECTION.value]

    # Store annotations that have been made so far.
    batch = Batch.from_json(batch_json)

    for idx, val in enumerate(batch.annotations):
        # Put samples that have not been to missing so they come up again.
        if val is None:
            batch.annotations[idx] = MISSING_LABEL

    completed_batch(dataset_id, batch, embedding_id)

    return dict(
        query_trigger=True,
        session_data=session_data,
    )


@callback(
    Input('back-button', 'n_clicks'),
    State('session-store', 'data'),
    State('batch-size-input', 'value'),
    State(ANNOT_PROGRESS, 'data'),
    output=dict(
        session_data=Output('session-store', 'data'),
        ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True),
        annot_progress=Output(ANNOT_PROGRESS, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_back_clicked(
    clicks,
    session_data,
    batch_size,
    annot_progress,
):
    if clicks is None:
        raise PreventUpdate

    print("on back click callback")
    batch = Batch.from_json(session_data[StoreKey.BATCH_STATE.value])

    if batch.progress == 0:
        print("Have to get last batch to be able to go back.")
        activeml_cfg = compose_from_state(session_data)

        # TODO it could be there is not enough labeled samples or there is no labeled samples anymore!
        batch, num_annotations = undo_annots_and_restore_batch(activeml_cfg, batch_size)
        # Decrease amount of annotations

        if batch is None:
            # There is no annotations anymore.
            logging.warning("Cannot go back further. No Annotations")
            return util.no_update()
        else:
            annot_progress[AnnotProgress.PROGRESS.value] = num_annotations

    batch.progress -= 1
    session_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    return dict(
        ui_trigger=True,
        session_data=session_data,
        annot_progress=annot_progress
    )


@callback(
    Input(UI_TRIGGER, 'data'),
    State(ANNOT_PROGRESS, 'data'),
    output=dict(
        annot_progress=Output(ANNOT_PROGRESS_TEXT, 'value'),
        num_samples=Output(NUM_SAMPLES_TEXT, 'value'),
    ),
    prevent_initial_call=True
)
def on_annot_progress(
    trigger,
    annot_data
):
    if trigger is None:
        raise PreventUpdate

    return dict(
        annot_progress=annot_data.get(AnnotProgress.PROGRESS.value),
        num_samples=annot_data.get(AnnotProgress.TOTAL_NUM.value)
    )


clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToChip'),
    Output("label-radio", 'value'),
    Input("label-search-text-input", "value"),
    prevent_initial_call=True,
)


