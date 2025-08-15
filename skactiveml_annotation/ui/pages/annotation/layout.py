import logging
from datetime import datetime

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
import dash_loading_spinners as dls

from hydra.utils import instantiate

# TODO: Change import style
from skactiveml_annotation.ui.common import compose_from_state
from skactiveml_annotation.ui.pages.annotation.auto_annotate_modal import create_auto_annotate_modal
from skactiveml_annotation.ui.pages.annotation.data_display_modal import create_data_display_modal
from skactiveml_annotation.ui.pages.annotation.label_setting_modal import create_label_settings_modal
from skactiveml_annotation.core.api import (
    request_query,
    completed_batch,
    load_embeddings,
    get_file_paths,
    get_total_num_samples,
    get_num_annotated,
    save_partial_annotations,
    add_class,
)
from skactiveml_annotation.core.api import undo_annots_and_restore_batch

from skactiveml_annotation.core.schema import *
from skactiveml_annotation.ui.storekey import StoreKey, AnnotProgress

from skactiveml_annotation.ui.pages.annotation.ids import *
from skactiveml_annotation.ui.pages.annotation.components import *
from skactiveml_annotation.ui.pages.annotation.data_display import *

register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)


def advance_batch(batch: Batch, annotation):
    idx = batch.progress
    batch.annotations[idx] = annotation
    batch.progress += 1


def layout(**kwargs):
    return(
        dmc.Box(
            [
                dcc.Location(id='url-annotation', refresh=True),
                dcc.Location(id=ANNOTATION_INIT, refresh=False),
                dcc.Store(id=UI_TRIGGER),
                dcc.Store(id=QUERY_TRIGGER),
                dcc.Store(id=START_TIME_TRIGGER),
                dcc.Store(id=ANNOT_PROGRESS, storage_type='session'),
                dcc.Store(id=ADD_CLASS_INSERTION_IDXES, storage_type='session'),
                dcc.Store(id=ADD_CLASS_WAS_ADDED, storage_type='session', data=False),

                create_label_settings_modal(),
                create_data_display_modal(),
                create_auto_annotate_modal(),

                dmc.Box(id='label-radio'),  # avoid id error
                dmc.AppShell(
                    [
                        dmc.AppShellNavbar(
                            id="sidebar-container-annotation",
                            children=create_sidebar(),
                            p="md",
                            # style={'border': '4px solid red'}
                        ),


                        dmc.Flex(
                            [
                                dmc.Box(
                                    [
                                        dmc.LoadingOverlay(
                                            id=COMPUTING_OVERLAY,
                                            zIndex=10,
                                            loaderProps=dict(
                                                children=dmc.Stack(
                                                    [
                                                        dmc.Group(
                                                            [
                                                                dmc.Title("Computing next batch", order=2),
                                                                dmc.Loader(
                                                                    size='xl',
                                                                    type='dots',
                                                                    color='var(--mantine-color-dark-7)',
                                                                ),
                                                            ],
                                                            justify='center',
                                                            wrap='wrap',
                                                            mb='5vh'
                                                        ),
                                                    ],
                                                    align='center',
                                                    # style=dict(border='red dashed 3px')
                                                )
                                            ),
                                            overlayProps=dict(
                                                radius='lg',
                                                center=True,
                                                blur=7
                                            ),
                                            transitionProps=dict(
                                                transition='fade',
                                                duration=150,
                                                # exitDuration=500,
                                            ),
                                        ),

                                        dmc.Center(
                                            dcc.Loading(
                                                dmc.Box(
                                                    id=DATA_DISPLAY_CONTAINER,
                                                    w='250px',
                                                    h='250px',
                                                    my=10,
                                                    # style=dict(border='4px dotted red')
                                                ),
                                                delay_hide=150,
                                                delay_show=150,
                                                custom_spinner=dls.ThreeDots(radius=7)
                                            ),
                                        ),

                                        dmc.Group(
                                            [
                                                dmc.Tooltip(
                                                    dmc.ActionIcon(
                                                        DashIconify(icon='tabler:plus',width=20),
                                                        variant='filled',
                                                        id=ADD_CLASS_BTN,
                                                        color="dark"
                                                    ),
                                                    label="Add a new class by using current Search Input."
                                                ),

                                                dmc.Tooltip(
                                                    dmc.ActionIcon(
                                                        DashIconify(icon="clarity:settings-line", width=20),
                                                        variant="filled",
                                                        id=LABEL_SETTING_BTN,
                                                        color='dark',
                                                    ),
                                                    label='Label settings',
                                                ),

                                                create_confirm_buttons(),

                                                dmc.TextInput(
                                                    placeholder='Select Label',
                                                    id=LABEL_SEARCH_INPUT,
                                                    radius='sm',
                                                    w='150px',
                                                ),
                                            ],
                                            mt=15,
                                            justify='center'
                                        ),
                                    ],
                                    p='10px',
                                    pos="relative",
                                ),

                                dmc.Stack(
                                    id=LABELS_CONTAINER,
                                    # h='400px'
                                    align='center'
                                ),
                                # create_confirm_buttons(),
                                create_progress_bar()
                            ],


                            style={
                                # 'border': '5px dotted blue',
                                'height': '100%',
                                'widht': '100%',
                            },
                            justify='center',
                            align='center',
                            direction='column',
                            wrap='nowrap',
                            gap='10px',
                            py=0,
                            px=150,
                        ),

                        dmc.AppShellAside(
                            children=[
                                dmc.Stack(
                                    [
                                        dmc.Card(
                                            dmc.Stack(
                                                [
                                                    dmc.Center(
                                                        dmc.Title("Stats", order=3)
                                                    ),

                                                    dmc.Tooltip(
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
                                                        label="Number of samples annotated."
                                                    ),

                                                    dmc.Tooltip(
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
                                                        ),
                                                        label='Total number of samples in dataset'
                                                    )

                                                ],
                                                gap=5
                                            )
                                        )
                                    ],
                                    # p='xs',
                                    # style={'border': '3px dashed green'},
                                    align='center'
                                )
                            ],
                            p="xs",
                            # style={'border': '4px solid red'}
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
                # 'border': 'green dotted 5px'
            }
        )
    )


# Get initial browser config like dpr.
clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='getDpr'),
    Output('browser-data', 'data'),
    Input(ANNOTATION_INIT, 'pathname')
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

    if (
        batch_json is None or
        Batch.from_json(batch_json).is_completed()
    ):
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
        annot_data=Output(ANNOT_PROGRESS, 'data', allow_duplicate=True),
        search_text=Output(LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
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
    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])

    if trigger_id == 'skip-button':
        annotation = MISSING_LABEL_MARKER
    else:
        now_str = datetime.now().time().isoformat(timespec="milliseconds")
        batch.end_times[batch.progress] = now_str
        annotation = value if trigger_id == 'confirm-button' else DISCARD_MARKER

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

    return dict(
        store_data=store_data,
        annot_data=annot_data,
        search_text='',
    )


# TODO there should be a seperate store for the BATCH
@callback(
    Input(UI_TRIGGER, 'data'),
    State('session-store', 'data'),
    State('browser-data', 'data'),
    State(ADD_CLASS_WAS_ADDED, 'data'),
    State(ADD_CLASS_INSERTION_IDXES, 'data'),
    State(LABEL_SETTING_SHOW_PROBAS, 'checked'),
    State(LABEL_SETTING_SORTBY, 'value'),
    output=dict(
        label_container=Output(LABELS_CONTAINER, 'children'),
        show_container=Output(DATA_DISPLAY_CONTAINER, 'children'),
        batch_progress=Output('batch-progress-bar', 'value'),
        is_computing_overlay=Output(COMPUTING_OVERLAY, 'visible', allow_duplicate=True),
        data_width=Output(DATA_DISPLAY_CONTAINER, 'w'),
        data_height=Output(DATA_DISPLAY_CONTAINER, 'h'),
        annot_start_time_trigger=Output(START_TIME_TRIGGER, 'data'),
        was_class_added=Output(ADD_CLASS_WAS_ADDED, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True,
)
def on_ui_update(
    ui_trigger,
    store_data,
    browser_dpr,
    # Adding classes
    was_class_added,
    insertion_idxes,
    # Label settings
    show_probas,  # TODO this is confusing
    sort_by
):
    if ui_trigger is None and browser_dpr is None:
        raise PreventUpdate

    activeml_cfg = compose_from_state(store_data)
    print(activeml_cfg.dataset.data_type)
    data_type: DataType = instantiate(activeml_cfg.dataset.data_type)

    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    idx = batch.progress
    embedding_idx = batch.emb_indices[idx]

    human_data_path = get_file_paths(
        activeml_cfg.dataset.id,
        activeml_cfg.embedding.id,
        embedding_idx
    )

    rendered_data, w, h = create_data_display(data_type, human_data_path, browser_dpr)

    return dict(
        label_container=create_label_chips(activeml_cfg.dataset.classes, batch, show_probas, sort_by,
                                           was_class_added, insertion_idxes),
        show_container=rendered_data,
        batch_progress=(idx / len(batch.emb_indices)) * 100,
        is_computing_overlay=False,
        data_width=w,
        data_height=h,
        annot_start_time_trigger=True,
        was_class_added=False
    )


# On Query start. Show loading overlay.
clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='triggerTrue'),
    Output(COMPUTING_OVERLAY, 'visible'),
    Input(QUERY_TRIGGER, 'data')
)


@callback(
    Input(QUERY_TRIGGER, 'data'),
    State('session-store', 'data'),
    State('batch-size-input', 'value'),
    State('subsampling-input', 'value'),
    output=dict(
        store_data=Output('session-store', 'data', allow_duplicate=True),
        ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True),
        insertion_idxes=Output(ADD_CLASS_INSERTION_IDXES, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True,
    # background=True, # INFO LRU Cache won't work with this
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
    # TODO bad name. Sampling parameters
    session_cfg = SessionConfig(batch_size, subsampling)

    batch = request_query(activeml_cfg, session_cfg, X)
    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    return dict(
        store_data=store_data,
        ui_trigger=True,
        insertion_idxes=None,
    )


@callback(
    Input('skip-batch-button', 'n_clicks'),
    State('session-store', 'data'),
    State(ANNOT_PROGRESS, 'data'),
    output=dict(
        query_trigger=Output(QUERY_TRIGGER, 'data'),
        session_data=Output('session-store', 'data', allow_duplicate=True),
        annot_progress=Output(ANNOT_PROGRESS, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_skip_batch(
    n_clicks: int,
    session_data: dict,
    annot_progress,
):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    # reset batch state
    batch_json = session_data.pop(StoreKey.BATCH_STATE.value, None)
    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
    embedding_id = session_data[StoreKey.EMBEDDING_SELECTION.value]
    batch = Batch.from_json(batch_json)

    # TODO this should not be necessary
    num_annotated = save_partial_annotations(batch, dataset_id, embedding_id)
    annot_progress[AnnotProgress.PROGRESS.value] = num_annotated

    return dict(
        query_trigger=True,
        session_data=session_data,
        annot_progress=annot_progress
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
            raise PreventUpdate
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
    Input(LABEL_SEARCH_INPUT, "value"),
    prevent_initial_call=True,
)


@callback(
    Input(START_TIME_TRIGGER, 'data'),
    State('session-store', 'data'),
    output=dict(
        session_data=Output("session-store", 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_annot_start_timestamp(
    trigger,
    session_data
):
    if trigger is None:
        raise PreventUpdate

    batch = Batch.from_json(session_data[StoreKey.BATCH_STATE.value])
    # Problem that shit runs before ui rendering is complete.
    idx = batch.progress
    now_str = datetime.now().time().isoformat(timespec="milliseconds")
    batch.start_times[idx] = now_str
    session_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    return dict(
        session_data=session_data
    )


@callback(
    Input(ADD_CLASS_BTN, 'n_clicks'),
    State('session-store', 'data'),
    State(LABEL_SEARCH_INPUT, 'value'),
    State(ADD_CLASS_INSERTION_IDXES, 'data'),
    output=dict(
        # ui_trigger=Output(QUERY_TRIGGER, 'data', allow_duplicate=True),
        ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True),
        # search_value=Output(LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
        insertion_idxes=Output(ADD_CLASS_INSERTION_IDXES, 'data'),
        label_value=Output('label-radio', 'value', allow_duplicate=True),
        was_class_added=Output(ADD_CLASS_WAS_ADDED, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_add_new_class(
    click,
    session_data,
    new_class_name,
    insertion_idxes
):
    if click is None or new_class_name is None:
        raise PreventUpdate

    activeml_cfg = compose_from_state(session_data)

    insertion_idx = add_class(
        dataset_cfg=activeml_cfg.dataset,
        new_class_name=new_class_name
    )

    # Let UI know that there have been added some class for which no probas will exist
    # before refitting with new classes
    if insertion_idxes is None:
        insertion_idxes = [insertion_idx]
    else:
        insertion_idxes.append(insertion_idx)

    return dict(
        ui_trigger=True,
        # search_value='',
        insertion_idxes=insertion_idxes,
        label_value=new_class_name,
        was_class_added=True,
    )
