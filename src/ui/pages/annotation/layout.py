import dash
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
    set_props
)
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

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

from paths import ROOT_PATH

from ui.pages.annotation.ids import *
from ui.pages.annotation.components import *
from ui.pages.annotation.data_display import *

register_page(
    __name__,
    path_template='/annotation/<dataset_name>',
    description='The main annotation page',
)

# TODO create variables for the id's


def id_placeholder():
    return dmc.Box(
        [
            dmc.Box(id='label-radio'),  # avoid id error
        ]
    )


def compose_from_state(store_data) -> ActiveMlConfig:
    overrides = (
        ('dataset', store_data[StoreKey.DATASET_SELECTION.value]),
        ('query_strategy', store_data[StoreKey.QUERY_SELECTION.value]),
        ('embedding', store_data[StoreKey.EMBEDDING_SELECTION.value]),
        ('+model', store_data[StoreKey.MODEL_SELECTION.value])  # add model to default list
    )

    # overrides = {
    #     'dataset': store_data[StoreKey.DATASET_SELECTION.value],
    #     'query_strategy': store_data[StoreKey.QUERY_SELECTION.value],
    #     'embedding': store_data[StoreKey.EMBEDDING_SELECTION.value],
    #     '+model': store_data[StoreKey.MODEL_SELECTION.value]  # add model to default list
    # }
    return compose_config(overrides)


def advance_batch(batch: Batch, annotation):
    idx = batch.progress
    batch.annotations[idx] = annotation
    batch.progress += 1


def layout(**kwargs):
    # noinspection PyTypeChecker
    return (
        dmc.Box(
            [
                dcc.Location(id='url-annotation', refresh=True),
                dcc.Location(id=ANNOTATION_INIT, refresh=False),
                dcc.Store(id=UI_TRIGGER),
                dcc.Store(id=QUERY_TRIGGER),
                id_placeholder(),
                dcc.Store(id='last-batch-store', storage_type='session'),
                dmc.AppShell(
                    [
                        dmc.AppShellNavbar(
                            id="sidebar-container-annotation",
                            children=create_sidebar(),
                            p="md",
                            style={'border': '4px solid red'}
                        ),
                        dmc.Box(
                            id="hero-container-annotation",
                            children=[

                                dmc.Box(
                                    html.Div("Main Content"),
                                    id=DATA_DISPLAY_CONTAINER
                                ),

                                # dcc.Loading(
                                #     children=[
                                #         dmc.Box(
                                #             html.Div("Main Content"),
                                #             id=DATA_DISPLAY_CONTAINER
                                #         ),
                                #     ],
                                #     delay_hide=1000,
                                #     custom_spinner=dmc.Skeleton(
                                #         visible=True,
                                #         h="100%"
                                #     )
                                # ),

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
                        "width": '12vw',
                        "breakpoint": "sm",
                        "collapsed": {"mobile": True},
                    },
                    aside={
                        "width": '12vw',
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


# region Next lvl
@callback(
    Input(ANNOTATION_INIT, 'pathname'),
    State('session-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def init(
    _,
    store_data,
):
    print("init callback")
    batch_json = store_data.get(StoreKey.BATCH_STATE.value)
    if batch_json is None:
        print("init trigger query")
        set_props(QUERY_TRIGGER, dict(data=True))
    else:
        # TODO need to deal with completed batch here aswell.
        print('init trigger ui')
        set_props(UI_TRIGGER, dict(data=True))


@callback(
    Input('confirm-button', 'n_clicks'),
    Input('discard-button', 'n_clicks'),
    Input('skip-button', 'n_clicks'),
    State('session-store', 'data'),
    State('label-radio', 'value'),
    output=dict(
        store_data=Output('session-store', 'data', allow_duplicate=True),
        # query_trigger=Output(QUERY_TRIGGER, 'data', allow_duplicate=True),
        # ui_trigger=Output(UI_TRIGGER, 'data', allow_duplicate=True),
        last_batch=Output('last-batch-store', 'data'),
    ),
    prevent_initial_call=True
)
def on_confirm(
    confirm_click,  # TODO use patter matching instead.
    discard_click,
    skip_click,
    store_data,
    value,
):
    if confirm_click is None and \
            discard_click is None and \
            skip_click is None:
        raise PreventUpdate

    print("\non_confirm callback")

    # ui_trigger = dash.no_update,
    # query_trigger = dash.no_update
    last_batch_json = dash.no_update

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
        # TODO maybe no trigger is needed here.
        print("BATCH IS COMPLETED")
        dataset_id = store_data[StoreKey.DATASET_SELECTION.value]
        completed_batch(dataset_id, batch)
        last_batch_json = batch.to_json()
        print('confirm triggers query')
        set_props(QUERY_TRIGGER, dict(data=True))
        # query_trigger = True
    else:
        # ui_trigger = True
        print('confirm triggers ui')
        set_props(UI_TRIGGER, dict(data=True))

    return dict(
        store_data=store_data,
        # query_trigger=query_trigger,
        # ui_trigger=ui_trigger,
        last_batch=last_batch_json,
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

    print('\non ui update')
    activeml_cfg = compose_from_state(store_data)
    X, file_names = get_embeddings(activeml_cfg)
    data_type: DataType = instantiate(activeml_cfg.dataset.data_type)

    batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
    idx = batch.progress
    query_idx = batch.indices[idx]
    human_data_path = file_names[query_idx]

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
    X, file_names = get_embeddings(activeml_cfg)
    session_cfg = SessionConfig(batch_size, subsampling)

    batch = request_query(activeml_cfg, session_cfg, X, file_names)
    store_data[StoreKey.BATCH_STATE.value] = batch.to_json()

    return dict(
        store_data=store_data,
        ui_trigger=True
    )

# endregion



# @callback(
#     Input(ANNOTATION_INIT, 'pathname'),
#     State('batch-size-input', 'value'),
#     State('subsampling-input', 'value'),
#     State('session-store', 'data'),
#     output=dict(
#         label_container=Output(LABELS_CONTAINER, 'children'),
#         show_container=Output(DATA_DISPLAY_CONTAINER, 'children'),
#         batch_progress=Output('batch-progres-bar', 'value'),
#         store_data=Output('session-store', 'data')
#     ),
#     # background=True,
#     prevent_initial_call=False
# )
# def init_page(
#     _,
#     batch_size,
#     subsampling,
#     store_data
# ):
#     print('init_page callback')
#
#     if subsampling == '':
#         subsampling = None
#     session_cfg = SessionConfig(batch_size=batch_size, subsampling=subsampling)
#
#     activeml_cfg = compose_from_state(store_data)
#     X, file_names = get_embeddings(activeml_cfg)
#
#     # TODO It could be that there is no batch yet then it has to be computed!
#     # batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
#     batch_json = store_data.get(StoreKey.BATCH_STATE.value)
#     # if batch_json is None:
#     #     batch = request_query(activeml_cfg, session_cfg, X, file_names)
#     #     store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
#     # else:
#     #     batch = Batch.from_json(batch_json)
#     batch = Batch.from_json(batch_json)
#
#     idx = batch.progress
#     query_idx = batch.indices[idx]
#     # TODO maybe the adapter should be responsible with specifying how to get human representation for sample with idx
#
#     data_type: DataType = instantiate(activeml_cfg.dataset.data_type)
#
#     human_data_path = file_names[query_idx]
#
#     return dict(
#         label_container=create_chip_group(activeml_cfg.dataset.classes, batch),
#         show_container=create_data_display(data_type, human_data_path),
#         batch_progress=idx / len(batch.indices),
#         store_data=store_data
#     )







# # TODO seperate Loading Image from
# # TODO rename this function to have a better name.
# def create_hero_section(classes: list[str], dataset_cfg: DatasetConfig, human_data_path: str, batch: Batch, progress: float):
#     # TODO instantiate the data_type enum somewhere else
#     data_type: DataType = instantiate(dataset_cfg.data_type)
#
#     human_data_path = ROOT_PATH / human_data_path
#
#     if data_type.value == DataType.IMAGE.value:
#         rendered_data = create_image_display(human_data_path)
#     elif data_type.value == DataType.TEXT.value:
#         rendered_data = create_text_display(human_data_path)
#     else:
#         rendered_data = create_audio_display(human_data_path)
#
#     class_prob = None
#     if batch.class_probas:
#         class_prob = batch.class_probas[batch.progress]
#
#     return (
#         # TODO this should be a Stack maybe. As I have to put a stack arround everything for no reason now.
#         dmc.Container(
#             [
#                 # Data display Container
#                 dmc.Stack(
#                     rendered_data,
#                     style={'border': '4px dotted pink'},
#                     align="center",
#                 ),
#
#                 # Label selection
#                 dmc.Group(
#                     [
#                         dmc.Title('Select Label', order=4),
#                         dmc.Tooltip(
#                             dmc.ActionIcon(
#                                 DashIconify(icon="clarity:settings-line", width=20),
#                                 variant="filled",
#                                 id="label-setting-popup",
#                             ),
#                             label='Label settings',
#                         ),
#                         dmc.TextInput(placeholder='Search', id='label-search-text-input'),
#                     ],
#                     justify='center'
#                 ),
#
#                 dmc.Stack(
#                     create_chip_group(classes, batch, class_prob),
#                     align='center'
#                 ),
#             ],
#             fluid=True,
#             style={
#                 'border': '4px dashed green',
#             },
#
#         )
#     )
#
#
# @callback(
#     Input('url-annotation', 'pathname'),
#     State('session-store', 'data'),
#     State('batch-size-input', 'value'),
#     State('subsampling-input', 'value'),
#     output=dict(
#         session_store=Output('session-store', 'data', allow_duplicate=True),
#         hero_container=Output('hero-container-annotation', 'children'),
#         last_batch=Output('last-batch-store', 'data', allow_duplicate=True)
#     ),
#     prevent_initial_call=True
# )
# def setup_annotations_page(
#         pathname,
#         store_data,
#         batch_size,
#         subsampling,
# ):
#     dataset_id = pathname.split('/')[-1]
#     print("[Annot] init annotation page with dataset: ", dataset_id)
#
#     if subsampling == '':
#         subsampling = None
#     session_cfg = SessionConfig(batch_size=batch_size, subsampling=subsampling)
#
#     # info overrides of lower lvl config can be done like so:
#     # cfg = compose(config_name="config", overrides=["database.host=remote_server"])
#
#     overrides = {
#         'dataset': store_data[StoreKey.DATASET_SELECTION.value],
#         'query_strategy': store_data[StoreKey.QUERY_SELECTION.value],
#         'embedding': store_data[StoreKey.EMBEDDING_SELECTION.value],
#         '+model': store_data[StoreKey.MODEL_SELECTION.value]  # add model to default list
#     }
#     print(overrides)
#
#     # TODO avoid reloading X and file_names every time. Cache could make sense.
#     activeMl_cfg = compose_config(overrides)
#     dataset_cfg = activeMl_cfg.dataset
#
#     # TODO This should only be needed when a batch is completed.
#     X, file_names = get_embeddings(activeMl_cfg)
#     last_batch_json = dash.no_update
#
#     if StoreKey.BATCH_STATE.value not in store_data:
#         # New Session
#         batch = request_query(activeMl_cfg, session_cfg, X, file_names)
#         store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
#     else:
#         # Existing Session
#         batch: Batch = Batch.from_json(store_data[StoreKey.BATCH_STATE.value])
#         if batch.is_completed():
#             print("BATCH IS COMPLETED")
#             # Store labeling data to disk
#             # TODO to much serialization deserialization
#
#             completed_batch(dataset_id, batch)
#
#             # Override last batch
#             last_batch_json = batch.to_json()
#             # Initialize the next batch
#             batch = request_query(activeMl_cfg, session_cfg, X, file_names)
#             store_data[StoreKey.BATCH_STATE.value] = batch.to_json()
#
#     print("Batch")
#     print('indices:', batch.indices)
#     print('progress', batch.progress)
#     print('annotations:', batch.annotations)
#
#     idx = batch.progress
#     # query_idx -> file_name
#     query_idx = batch.indices[idx]
#     progress_percent = idx / len(batch.indices)
#
#     # TODO generalize. How the human readable data and how the label names are fetched.
#     # From Cache?
#     classes = dataset_cfg.classes
#
#     # TODO maybe the adapter should be responsible with specifying how to get human representation for sample with idx
#     human_data_path = file_names[query_idx]
#     print('human data path')
#     print(human_data_path)
#
#     return dict(
#         session_store=store_data,
#         hero_container=create_hero_section(classes, dataset_cfg, human_data_path, batch, progress_percent),
#         last_batch=last_batch_json
#     )
#
#

# @callback(
#     Input('confirm-button', 'n_clicks'),
#     Input('discard-button', 'n_clicks'),
#     Input('skip-button', 'n_clicks'),
#     State('label-radio', 'value'),
#     State('session-store', 'data'),
#     output=dict(
#         session_data=Output('session-store', 'data', allow_duplicate=True),
#         pathname=Output(ANNOTATION_INIT, 'pathname'),
#     ),
#     prevent_initial_call=True,
# )
# def on_button_click(
#     confirm_clicks: int,
#     outlier_clicks: int,
#     skip_clicks: int,
#     value: int,
#     session_data: dict
# ):
#     # TODO is this needed?
#     if ((confirm_clicks is None or confirm_clicks == 0) and
#         (outlier_clicks is None or outlier_clicks == 0) and
#             (skip_clicks is None or skip_clicks == 0)):
#         raise PreventUpdate
#
#     trigger_id = callback_context.triggered_id
#     if trigger_id == "confirm-button":
#         annotation = int(value)
#     elif trigger_id == "skip-button":
#         annotation = MISSING_LABEL
#     else:
#         annotation = np.inf
#
#     print("Annotated label: ", annotation)
#
#     # Update the Session's Batch State
#     batch_state_json = session_data[StoreKey.BATCH_STATE.value]
#     batch: Batch = Batch.from_json(batch_state_json)
#     # print("pre", batch)
#     idx = batch.progress
#     batch.annotations[idx] = annotation
#     batch.progress += 1
#
#     # Override existing batch
#     session_data[StoreKey.BATCH_STATE.value] = batch.to_json()
#
#     # TODO the patch should be advanced here?
#
#     return dict(
#         session_data=session_data,
#         pathname=None  # Refresh page by passing None.
#     )

#
#
# @callback(
#     Input('skip-batch-button', 'n_clicks'),
#     State('session-store', 'data'),
#     output=dict(
#         session_data=Output('session-store', 'data', allow_duplicate=True),
#         pathname=Output('url-annotation', 'pathname', allow_duplicate=True)
#     ),
#     prevent_initial_call=True
# )
# def on_skip_batch(
#     n_clicks: int,
#     session_data: dict,
# ):
#     if n_clicks is None or n_clicks == 0:
#         raise PreventUpdate
#
#     # reset batch state
#     batch_json = session_data.pop(StoreKey.BATCH_STATE.value, None)
#     dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
#
#     # Store annotations that have been made so far.
#     batch = Batch.from_json(batch_json)
#
#     for idx, val in enumerate(batch.annotations):
#         if val is None:
#             batch.annotations[idx] = MISSING_LABEL
#
#     completed_batch(dataset_id, batch)
#
#     return dict(
#         session_data=session_data,
#         pathname=None  # By passing None the page is reloaded
#     )
#
#
# @callback(
#     Input('back-button', 'n_clicks'),
#     State('session-store', 'data'),
#     State('last-batch-store', 'data'),
#     output=dict(
#         session_data=Output('session-store', 'data'),
#         pathname=Output('url-annotation', 'pathname', allow_duplicate=True),
#         last_batch=Output('last-batch-store', 'data', allow_duplicate=True),
#     ),
#     prevent_initial_call=True
# )
# def on_back_clicked(
#     clicks,
#     session_data,
#     last_batch,
# ):
#     if clicks is None:
#         raise PreventUpdate
#
#     print("on back click callback")
#     batch = Batch.from_json(session_data[StoreKey.BATCH_STATE.value])
#
#     if batch.progress == 0:
#         print("Have to get last batch to be able to go back.")
#         if last_batch is None:
#             print("CANNOT go back further!")
#             raise PreventUpdate
#
#         batch = Batch.from_json(last_batch)
#         last_batch = None
#     else:
#         last_batch = dash.no_update
#
#     batch.progress -= 1
#     session_data[StoreKey.BATCH_STATE.value] = batch.to_json()
#     return dict(
#         session_data=session_data,
#         pathname=None,  # None to refresh the page.
#         last_batch=last_batch
#     )
#
#


clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToChip'),
    Output("label-radio", 'value'),
    Input("label-search-text-input", "value"),
    prevent_initial_call=True,
)


# TODO add cleanup function when switching away from annot page.


