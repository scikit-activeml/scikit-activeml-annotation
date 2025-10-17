import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import isodate

import dash
from dash import (
    dcc,
    register_page,
    callback,
    Input,
    Output,
    State,
    callback_context,
    clientside_callback,
    ClientsideFunction,
    set_props,
)
from dash.exceptions import PreventUpdate

from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_loading_spinners as dls

from skactiveml_annotation.core.data_display_model import DataDisplaySetting
from skactiveml_annotation.ui import common
from skactiveml_annotation.core import api

from skactiveml_annotation.core.schema import (
    Batch,
    Annotation,
    AnnotationList,
    SessionConfig,
    DISCARD_MARKER,
    MISSING_LABEL_MARKER,
)
from skactiveml_annotation.ui.storekey import StoreKey, AnnotProgress

from . import (
    ids,
    components,
    auto_annotate_modal,
    data_display_modal,
    label_setting_modal,
)
from .label_setting_modal import SortBySetting

register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)

def layout(**kwargs):
    _ = kwargs
    return(
        dmc.Box(
            [
                dcc.Location(id='url-annotation', refresh=True),
                dcc.Location(id=ids.ANNOTATION_INIT, refresh=False),
                # Triggers
                dcc.Store(id=ids.UI_TRIGGER),
                dcc.Store(id=ids.QUERY_TRIGGER),
                dcc.Store(id=ids.START_TIME_TRIGGER),
                # Data
                dcc.Store(id=ids.DATA_DISPLAY_CFG_DATA, storage_type='session'),
                dcc.Store(id=ids.ANNOT_PROGRESS, storage_type='session'),
                dcc.Store(id=ids.ADD_CLASS_INSERTION_IDXES, storage_type='session'),
                dcc.Store(id=ids.ADD_CLASS_WAS_ADDED, storage_type='session', data=False),

                label_setting_modal.create_label_settings_modal(),
                # TODO: this is a problem as it depends on the data type: text, image etc
                data_display_modal.create_data_display_modal(),

                auto_annotate_modal.create_auto_annotate_modal(),

                dmc.Box(id='label-radio'),  # avoid id error
                dmc.AppShell(
                    [
                        dmc.AppShellNavbar(
                            id="sidebar-container-annotation",
                            children=components.create_sidebar(),
                            p="md",
                            # style={'border': '4px solid red'}
                        ),
                        dmc.Flex(
                            [
                                dmc.Box(
                                    [
                                        dmc.LoadingOverlay(
                                            id=ids.COMPUTING_OVERLAY,
                                            zIndex=10,
                                            loaderProps={
                                                'children': dmc.Stack(
                                                    [
                                                        dmc.Group(
                                                            [
                                                                dmc.Title("Computing next batch", order=2),
                                                                # Show 3 dots duruing query
                                                                dmc.Loader(
                                                                    size='xl',
                                                                    type='dots',
                                                                    # Type annotation incorrect valid css is supported
                                                                    color='var(--mantine-color-dark-7)',  # pyright: ignore[reportArgumentType]
                                                                ),
                                                            ],
                                                            justify='center',
                                                            wrap='wrap',
                                                            mb='5vh'  # pyright: ignore[reportArgumentType]
                                                        ),
                                                    ],
                                                    align='center',
                                                    # style=dict(border='red dashed 3px')
                                                )
                                            },
                                            overlayProps={
                                                'radius':'lg',
                                                'center': True,
                                                'blur': 7,
                                            },
                                            transitionProps={
                                                'transition':'fade',
                                                'duration': 150,
                                                'mounted': True,
                                                # 'exitDuration': 500,
                                            },
                                        ),

                                        dmc.Center(
                                            dcc.Loading(
                                                dmc.Box(
                                                    id=ids.DATA_DISPLAY_CONTAINER,
                                                    # TODO why did I fix the width and heigh here?
                                                    mih=15,
                                                    # w='250px',
                                                    # h='250px',
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
                                                        id=ids.ADD_CLASS_BTN,
                                                        color="dark"
                                                    ),
                                                    label="Add a new class by using current Search Input."
                                                ),

                                                dmc.Tooltip(
                                                    dmc.ActionIcon(
                                                        DashIconify(icon="clarity:settings-line", width=20),
                                                        variant="filled",
                                                        id=ids.LABEL_SETTING_BTN,
                                                        color='dark',
                                                    ),
                                                    label='Label settings',
                                                ),

                                                components.create_confirm_buttons(),

                                                dmc.TextInput(
                                                    placeholder='Select Label',
                                                    id=ids.LABEL_SEARCH_INPUT,
                                                    radius='sm',
                                                    w='150px',
                                                ),
                                            ],
                                            mt=15,
                                            justify='center'
                                        ),
                                    ],
                                    p=10,
                                    pos="relative",
                                ),

                                dmc.Stack(
                                    id=ids.LABELS_CONTAINER,
                                    # h='400px'
                                    align='center'
                                ),
                                # create_confirm_buttons(),
                                components.create_progress_bar()
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
                            gap=10,
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
                                                                        id=ids.ANNOT_PROGRESS_TEXT,
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
                                                                        id=ids.NUM_SAMPLES_TEXT,
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
                    navbar={  # pyright: ignore[reportArgumentType]
                        "width": '13vw',
                        "breakpoint": "sm",
                        "collapsed": {"mobile": True},
                    },
                    aside={  # pyright: ignore[reportArgumentType]
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
    Input(ids.ANNOTATION_INIT, 'pathname')
)


@callback(
    Input(ids.ANNOTATION_INIT, 'pathname'),
    State('session-store', 'data'),
    output=dict(
        annot_progress=Output(ids.ANNOT_PROGRESS, 'data'),
        data_display_modal_children=Output(ids.DATA_DISPLAY_MODAL, 'children'),
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        query_trigger=Output(ids.QUERY_TRIGGER, 'data', allow_duplicate=True),
    ),
    prevent_initial_call='initial_duplicate'
)
def init(
    _,
    store_data,
):
    batch_json = store_data.get(StoreKey.BATCH_STATE.value)
    must_query = (
        batch_json is None or
        Batch.from_json(batch_json).is_completed()
    )
    
    if must_query:
        ui_trigger = dash.no_update
        query_trigger = True
    else:
        ui_trigger = True
        query_trigger = dash.no_update

    activeml_cfg = common.compose_from_state(store_data)
    data_type = activeml_cfg.dataset.data_type.instantiate()

    return dict(
        ui_trigger=ui_trigger,
        query_trigger=query_trigger,
        annot_progress=init_annot_progress(store_data),
        data_display_modal_children=data_display_modal.create_modal_content(data_type)
    )

def init_annot_progress(store_data):
    dataset_id = store_data.get(StoreKey.DATASET_SELECTION.value)
    embedding_id = store_data.get(StoreKey.EMBEDDING_SELECTION.value)

    return {
        AnnotProgress.PROGRESS.value: api.get_num_annotated(dataset_id),
        AnnotProgress.TOTAL_NUM.value: api.get_total_num_samples(dataset_id, embedding_id)
    }


@callback(
    Input('confirm-button', 'n_clicks'),
    Input('discard-button', 'n_clicks'),
    Input('skip-button', 'n_clicks'),
    State('session-store', 'data'),
    State('label-radio', 'value'),
    State(ids.ANNOT_PROGRESS, 'data'),
    output=dict(
        store_data=Output('session-store', 'data', allow_duplicate=True),
        annot_data=Output(ids.ANNOT_PROGRESS, 'data', allow_duplicate=True),
        search_text=Output(ids.LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
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
    if trigger_id == 'skip-button':
        label = MISSING_LABEL_MARKER
    else:
        label = value if trigger_id == 'confirm-button' else DISCARD_MARKER

    # Take timestamp when annotation was finished
    end_time = datetime.now()
    start_time = datetime.fromisoformat(store_data[StoreKey.DATA_PRESENT_TIMESTAMP.value])
    delta_time = end_time - start_time

    # TODO compute view duration and add to existing view duration
    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    idx = batch.progress
    # TODO that seems wierd
    annotations_list = AnnotationList.model_validate(store_data[StoreKey.ANNOTATIONS_STATE.value])
    annotations = annotations_list.annotations
    old_annotation = annotations[idx]

    if old_annotation is None:
        # New Annotation
        first_view_time = start_time.isoformat()
        total_view_duration = isodate.duration_isoformat(delta_time)
    else:
        # Sample was annotated before update the annotation
        # TODO only store last edit when there was a change?
        old_delta_time = isodate.parse_duration(old_annotation.total_view_duration)
        delta_time += old_delta_time
        first_view_time = old_annotation.first_view_time
        total_view_duration = isodate.duration_isoformat(delta_time)

    annotation = Annotation(
        embedding_idx=batch.emb_indices[idx],
        label=label,
        first_view_time=first_view_time,
        total_view_duration=total_view_duration,
        last_edit_time=end_time.isoformat(),
    )

    # TODO write helper for this in api?
    annotations[idx] = annotation

    batch.advance(step=1)

    # Override existing batch
    # TODO serialize here?
    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    store_data[StoreKey.ANNOTATIONS_STATE.value] = annotations_list.model_dump()

    if batch.is_completed():
        dataset_id = store_data[StoreKey.DATASET_SELECTION.value]
        embedding_id = store_data[StoreKey.EMBEDDING_SELECTION.value]

        # All samples in the batch should be annotated by now
        annotations = cast(list[Annotation], annotations)
        num_annotated = api.completed_batch(dataset_id, embedding_id, annotations, batch)
        if num_annotated == annot_data[AnnotProgress.TOTAL_NUM.value]:
            print("ANNOTATION COMPLETE")
            raise PreventUpdate

        annot_data[AnnotProgress.PROGRESS.value] = num_annotated

        set_props(ids.QUERY_TRIGGER, dict(data=True))
    else:
        set_props(ids.UI_TRIGGER, dict(data=True))

    return dict(
        store_data=store_data,
        annot_data=annot_data,
        search_text='',
    )


# TODO there should be a seperate store for the BATCH
@callback(
    Input(ids.UI_TRIGGER, 'data'),
    State('session-store', 'data'),
    State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
    State('browser-data', 'data'),
    State(ids.ADD_CLASS_WAS_ADDED, 'data'),
    State(ids.ADD_CLASS_INSERTION_IDXES, 'data'),
    State(ids.LABEL_SETTING_SHOW_PROBAS, 'checked'),
    State(ids.LABEL_SETTING_SORTBY, 'value'),
    output=dict(
        label_container=Output(ids.LABELS_CONTAINER, 'children'),
        show_container=Output(ids.DATA_DISPLAY_CONTAINER, 'children'),
        batch_progress=Output('batch-progress-bar', 'value'),
        data_display_data=Output(ids.DATA_DISPLAY_CFG_DATA, 'data'),
        is_computing_overlay=Output(ids.COMPUTING_OVERLAY, 'visible', allow_duplicate=True),
        data_width=Output(ids.DATA_DISPLAY_CONTAINER, 'w'),
        data_height=Output(ids.DATA_DISPLAY_CONTAINER, 'h'),
        annot_start_time_trigger=Output(ids.START_TIME_TRIGGER, 'data'),
        was_class_added=Output(ids.ADD_CLASS_WAS_ADDED, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True,
)
def on_ui_update(
    # Triggers
    ui_trigger,
    # Data
    store_data,
    data_display_setting,
    browser_dpr,
    # Adding classes
    was_class_added,
    insertion_idxes,
    # Label settings
    show_probas: bool,  # TODO this is confusing
    sort_by: str,
):
    if ui_trigger is None and browser_dpr is None:
        raise PreventUpdate

    activeml_cfg = common.compose_from_state(store_data)
    data_type = activeml_cfg.dataset.data_type.instantiate()

    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    annotations_list = AnnotationList.model_validate(store_data[StoreKey.ANNOTATIONS_STATE.value])
    
    idx = batch.progress
    embedding_idx = batch.emb_indices[idx]
    annotation = annotations_list.annotations[idx]

    human_data_path = Path(
        api.get_one_file_path(
            activeml_cfg.dataset.id,
            activeml_cfg.embedding.id,
            embedding_idx
        ) 
    )

    print(human_data_path)

    if data_display_setting is None:
        logging.info("Data Display Setting is not yet initialized. Initializing now.")
        data_display_setting = DataDisplaySetting()
    else:
        data_display_setting = DataDisplaySetting.model_validate(data_display_setting)
    rendered_data, w, h = components.create_data_display(data_display_setting, data_type, human_data_path, browser_dpr)

    sort_by = SortBySetting[sort_by] 

    # TODO how to organize this better?
    return dict(
        label_container=components.create_label_chips(
            activeml_cfg.dataset.classes, annotation, batch, show_probas, sort_by,
            was_class_added, insertion_idxes
        ),
        show_container=rendered_data,
        batch_progress=(idx / len(batch.emb_indices)) * 100,
        data_display_data=data_display_setting.model_dump(),
        is_computing_overlay=False,
        data_width=w,
        data_height=h,
        annot_start_time_trigger=True,
        was_class_added=False
    )


# On Query start. Show loading overlay.
clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='triggerTrue'),
    Output(ids.COMPUTING_OVERLAY, 'visible'),
    Input(ids.QUERY_TRIGGER, 'data')
)


@callback(
    Input(ids.QUERY_TRIGGER, 'data'),
    State('session-store', 'data'),
    State('batch-size-input', 'value'),
    State('subsampling-input', 'value'),
    output=dict(
        store_data=Output('session-store', 'data', allow_duplicate=True),
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        insertion_idxes=Output(ids.ADD_CLASS_INSERTION_IDXES, 'data', allow_duplicate=True)
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
    activeml_cfg = common.compose_from_state(store_data)
    X = api.load_embeddings(activeml_cfg.dataset.id, activeml_cfg.embedding.id)
    # TODO bad name. Sampling parameters
    session_cfg = SessionConfig(batch_size, subsampling)

    batch, annotations_list  = api.request_query(activeml_cfg, session_cfg, X)
    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    store_data[StoreKey.ANNOTATIONS_STATE.value] = annotations_list.model_dump()

    return dict(
        store_data=store_data,
        ui_trigger=True,
        insertion_idxes=None,
    )


@callback(
    Input('skip-batch-button', 'n_clicks'),
    State('session-store', 'data'),
    State(ids.ANNOT_PROGRESS, 'data'),
    output=dict(
        query_trigger=Output(ids.QUERY_TRIGGER, 'data'),
        session_data=Output('session-store', 'data', allow_duplicate=True),
        annot_progress=Output(ids.ANNOT_PROGRESS, 'data', allow_duplicate=True)
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
    num_annotated = api.save_partial_annotations(batch, dataset_id, embedding_id)
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
    State(ids.ANNOT_PROGRESS, 'data'),
    output=dict(
        session_data=Output('session-store', 'data'),
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        annot_progress=Output(ids.ANNOT_PROGRESS, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_back_clicked(
    clicks,
    store_data,
    batch_size,
    annot_progress,
):
    if clicks is None:
        raise PreventUpdate

    print("on back click callback")
    # TODO only store last edit when there was a change?

    # TODO repeated code create helper for this.
    end_time = datetime.now()
    start_time = datetime.fromisoformat(store_data[StoreKey.DATA_PRESENT_TIMESTAMP.value])
    delta_time = end_time - start_time

    # TODO compute view duration and add to existing view duration
    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    idx = batch.progress
    # TODO that seems wierd
    annotations_list = AnnotationList.model_validate(store_data[StoreKey.ANNOTATIONS_STATE.value])
    annotations = annotations_list.annotations
    old_annotation = annotations[idx]

    if old_annotation is None:
        # TODO should not happen right?
        print("Does this happen?")
        first_view_time = start_time.isoformat()
        total_view_duration = isodate.duration_isoformat(delta_time)
        label = MISSING_LABEL_MARKER
    else:
        # Sample was annotated before update the annotation
        # TODO only store last edit when there was a change?
        old_delta_time = isodate.parse_duration(old_annotation.total_view_duration)
        delta_time += old_delta_time
        first_view_time = old_annotation.first_view_time
        total_view_duration = isodate.duration_isoformat(delta_time)
        label = old_annotation.label

    annotation = Annotation(
        embedding_idx=batch.emb_indices[idx],
        label=label,
        first_view_time=first_view_time,
        total_view_duration=total_view_duration,
        last_edit_time=end_time.isoformat(),
    )

    # TODO write helper for this in api?
    annotations[idx] = annotation

    if batch.progress == 0:
        # TODO write helper for that
        print("Have to get last batch to be able to go back.")
        # TODO Serialize new Annotations
        dataset_id = store_data.get(StoreKey.DATASET_SELECTION.value)
        embedding_id = store_data.get(StoreKey.EMBEDDING_SELECTION.value)

        # TODO i should only get the file paths for the samples acctually annotated
        file_paths = api.get_file_paths(dataset_id, embedding_id, batch.emb_indices)

        _ = api.update_annotations(dataset_id, file_paths, annotations)

        # TODO this step should be done in the serialize and deserialize methods
        activeml_cfg = common.compose_from_state(store_data)

        try:
            batch, annotations_list, num_annotations = api.undo_annots_and_restore_batch(activeml_cfg, batch_size)
            # TODO should annotations be decreased when going back?
            # Decrease amount of annotations
            annot_progress[AnnotProgress.PROGRESS.value] = num_annotations
        except RuntimeError:
            logging.warning("Cannot go back further. No Annotations")
            raise PreventUpdate
    else:
         batch.advance(step= -1)

    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
    store_data[StoreKey.ANNOTATIONS_STATE.value] = annotations_list.model_dump()
    return dict(
        ui_trigger=True,
        session_data=store_data,
        annot_progress=annot_progress
    )


@callback(
    Input(ids.UI_TRIGGER, 'data'),
    State(ids.ANNOT_PROGRESS, 'data'),
    output=dict(
        annot_progress=Output(ids.ANNOT_PROGRESS_TEXT, 'value'),
        num_samples=Output(ids.NUM_SAMPLES_TEXT, 'value'),
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
    Input(ids.LABEL_SEARCH_INPUT, "value"),
    prevent_initial_call=True,
)


@callback(
    Input(ids.START_TIME_TRIGGER, 'data'),
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

    # TODO Problem that shit runs before ui rendering is complete.
    now_str = datetime.now().isoformat()

    session_data[StoreKey.DATA_PRESENT_TIMESTAMP.value] = now_str

    return dict(
        session_data=session_data
    )


@callback(
    Input(ids.ADD_CLASS_BTN, 'n_clicks'),
    State('session-store', 'data'),
    State(ids.LABEL_SEARCH_INPUT, 'value'),
    State(ids.ADD_CLASS_INSERTION_IDXES, 'data'),
    output=dict(
        # ui_trigger=Output(QUERY_TRIGGER, 'data', allow_duplicate=True),
        ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
        # search_value=Output(LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
        insertion_idxes=Output(ids.ADD_CLASS_INSERTION_IDXES, 'data'),
        label_value=Output('label-radio', 'value', allow_duplicate=True),
        was_class_added=Output(ids.ADD_CLASS_WAS_ADDED, 'data', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def on_add_new_class(
    click,
    session_data,
    new_class_name,
    insertion_idxes: list[int]
):
    if click is None or new_class_name is None:
        raise PreventUpdate

    activeml_cfg = common.compose_from_state(session_data)

    insertion_idx = api.add_class(
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
