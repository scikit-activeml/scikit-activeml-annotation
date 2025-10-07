from dash import (
    dcc,
    callback,
    clientside_callback,
    ClientsideFunction,
    Input,
    Output,
    State,
    html,
    register_page,
)
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc
from dash_iconify import DashIconify

from skactiveml_annotation.core import api
from skactiveml_annotation.core.schema import DatasetConfig
from skactiveml_annotation.ui.components import sampling_input
from skactiveml_annotation.ui.storekey import StoreKey

RADIO_SELECTION = 'radio-selection'

register_page(__name__, path='/')


def layout(**kwargs: object):
    _ = kwargs
    return (
        dmc.Center(
            [
                dcc.Location(id='url_home', refresh=True),
                dcc.Location(id='url_home_init', refresh=False),
                dmc.Stack(
                    [
                        dmc.Stack(
                            [
                                dmc.Title("Welcome to scikit-activeml-annotation", order=1),
                                dmc.Title("Configure your annotation pipeline", order=2)
                            ],
                            align='center',
                            p='xl'
                        ),

                        dmc.Flex(
                            [
                                create_stepper(),
                                dcc.Loading(
                                    dmc.Container(
                                        # Current selection injected here
                                        html.Div(id=RADIO_SELECTION),  # workaround so id exists at the start
                                        id='selection_container',
                                        # TODO use Mantine styling for this.
                                        style={
                                            "min-width": "15vw",
                                            "whiteSpace": "normal", "wordWrap": "break-word",
                                            # 'border': '3px dotted red'
                                        }
                                    ),
                                    type='circle',
                                    delay_hide=150,
                                    delay_show=250  # INFO only need to show when it takes longer than 5ms.
                                )
                            ],
                            gap='xl',
                            justify='center',
                            align='flex-start',
                        ),

                        dmc.Group(
                            [
                                dmc.Button("Back", id='back_button', color='dark'),
                                dmc.Button("Confirm", id='confirm_button', color='dark', disabled=True)
                            ]
                        )
                    ],
                    align='center',
                    # style={'border': '2px solid gold'}
                )
            ],
            mt=1,
            style={'height': '100%'}
        )
    )


# Helper function to build UI for different steps
def create_step_ui(step: int, session_data):
    if step == 0:
        if session_data is None:
            preselect = None
        else:
            preselect = session_data.get(StoreKey.DATASET_SELECTION.value)
        content = _create_dataset_selection(preselect)
    elif step == 1:
        content = _create_embedding_radio_group(session_data)
    elif step == 2:
        content = _create_radio_group(api.get_qs_config_options(), session_data.get(StoreKey.QUERY_SELECTION.value))
    elif step == 3:
        content = _create_radio_group(api.get_model_config_options(), session_data.get(StoreKey.MODEL_SELECTION.value))
    elif step == 4:
        content = dmc.Stack(
            [
                *sampling_input.create_sampling_inputs(),
                # Dummy element to ensure this id exists in the layout at the last step
                dmc.RadioGroup([], id=RADIO_SELECTION, display='none', readOnly=True)
            ]
        )
    elif step == 5:
        return None
    else:
        raise RuntimeError("Step is not in {0,...,4}")

    return dmc.ScrollArea(
        content,
        offsetScrollbars='y',
        type='auto',
        scrollbars='y',
        styles=dict(
            viewport={
                'maxHeight': '100%',
            }
        )
    )


def create_stepper():
    return (
        dmc.Stepper(
            [
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 0}, label="Dataset", description="Select a Dataset"),  # loading=True),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 1}, label="Embedding", description="Select embedding method"),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 2}, label="Query Strategy", description="Select a Query Strategy"),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 3}, label="Model", description="Select a model"),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 4}, label="Sampling", description="Set Sampling parameters"),
            ],
            id='stepper',
            active=0,
            orientation='vertical',
            iconSize=40,
            size='xl',
            # style={'border': '2px solid red'},
            allowNextStepsSelect=False
        )
    )


def _create_dataset_radio_item(cfg: DatasetConfig, cfg_display: str):
    dataset_exists = api.dataset_path_exits(cfg.data_path)

    radio_item = (
        dmc.Radio(
            label=cfg_display,
            value=cfg.id,
            disabled=not dataset_exists,
            size='md'
        )
    )

    if dataset_exists:
        return radio_item

    return dmc.Tooltip(
        radio_item,
        label=f'Dataset does not exist at path: {cfg.data_path}',
        openDelay=250,
    )


def _create_dataset_selection(preselect):
    print("_create data selection invoked")
    dataset_options = api.get_dataset_config_options()
    data = [(cfg, f'{cfg.display_name} - ({cfg.data_type.instantiate().value})')
            for cfg in dataset_options]

    # TODO repeated code.
    return \
        dmc.RadioGroup(
            id=RADIO_SELECTION,
            children=dmc.Stack(
                [
                    _create_dataset_radio_item(cfg, cfg_display)
                    for cfg, cfg_display in data
                ]
            ),
            value=preselect,
            size="md",
            # style={'border': '2px solid red'}
        )


def _create_embedding_radio_group(session_data):
    # TODO only display embeddings that are valid for the selected dataset
    options = api.get_embedding_config_options()
    formatted_options = [(cfg.id, cfg.display_name) for cfg in options]

    preselect = session_data.get(StoreKey.EMBEDDING_SELECTION.value)

    return dmc.RadioGroup(
        id=RADIO_SELECTION,
        children=dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.Radio(label=cfg_name, value=cfg_id, size='md'),
                        _create_bool_icon(api.is_dataset_embedded(
                            session_data[StoreKey.DATASET_SELECTION.value],
                            cfg_id
                        ))
                    ]
                )
                for cfg_id, cfg_name in formatted_options
            ]
        ),
        value=preselect,
        size="md",
        # style={'border': '2px solid red'},
    )


def _create_bool_icon(val: bool):
    # TODO do this via CSS
    if val:
        icon = 'tabler:check'
        color = 'green'
        label = 'embedding is cached'
    else:
        icon = 'tabler:x'
        color = 'red'
        label = 'embedding has to be computed'

    return (
        dmc.Tooltip(
            dmc.ThemeIcon(
                DashIconify(icon=icon),
                variant='light',
                radius=20,
                color=color,
                size=25,
            ),
            label=label
        )
    )


# Helper function to create a radio group
def _create_radio_group(options, preselect):
    formatted_options = [(cfg.id, cfg.display_name) for cfg in options]
    return dmc.RadioGroup(
        id=RADIO_SELECTION,
        children=dmc.Stack([dmc.Radio(label=l, value=k, size='md') for k, l in formatted_options]),
        value=preselect,
        size="md",
        # style={'border': '2px solid red'},
    )
# endregion


@callback(
    Input('url_home_init', 'pathname'),
    State('session-store', 'data'),
    output=dict(
        selection_content=Output('selection_container', 'children', allow_duplicate=True),
        session_data=Output('session-store', 'data', allow_duplicate=True)
    ),
    prevent_initial_call='initial_duplicate'
)
def setup_page(
    _,
    session_data
):
    print("Setup page")

    if session_data is None:
        session_data = {}

    return dict(
        selection_content=create_step_ui(0, session_data),
        session_data=session_data
    )


@callback(
    Input('confirm_button', 'n_clicks'),
    State(RADIO_SELECTION, 'value'),
    State('stepper', 'active'),
    State('session-store', 'data'),
    output=dict(
        selection_content=Output('selection_container', 'children', allow_duplicate=True),
        session_data=Output('session-store', 'data', allow_duplicate=True),
        step=Output('stepper', 'active', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def handle_confirm(
    n_clicks,
    radio_value,
    current_step,
    session_data
):
    print(f"handle_confirm triggered at step {current_step} with radio_value: {radio_value}")
    if current_step >= 4 or radio_value is None or n_clicks is None:
        raise PreventUpdate

    if current_step == 0:
        # TODO move this somewhere else. This is bad here.
        prev_dataset_id = session_data.get(StoreKey.DATASET_SELECTION.value)
        was_dataset_changed = prev_dataset_id is not None and radio_value != prev_dataset_id
        if was_dataset_changed:
            session_data.pop(StoreKey.BATCH_STATE.value, None)

        session_data[StoreKey.DATASET_SELECTION.value] = radio_value

    elif current_step == 1:
        session_data[StoreKey.EMBEDDING_SELECTION.value] = radio_value

    elif current_step == 2:
        session_data[StoreKey.QUERY_SELECTION.value] = radio_value

    elif current_step == 3:
        session_data[StoreKey.MODEL_SELECTION.value] = radio_value

    new_step = current_step + 1
    return dict(
        selection_content=create_step_ui(new_step, session_data),
        session_data=session_data,
        step=new_step
    )


# Back button callback
@callback(
    Input('back_button', 'n_clicks'),
    State('stepper', 'active'),
    State('session-store', 'data'),
    output=dict(
        children=Output('selection_container', 'children', allow_duplicate=True),
        active=Output('stepper', 'active', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def handle_back(
    _,
    current_step,
    session_data
):
    print("handle_back callback")
    if current_step == 0:
        raise PreventUpdate

    next_step = current_step - 1

    return dict(
        children=create_step_ui(next_step, session_data),
        active=next_step
    )


@callback(
    Input('confirm_button', 'n_clicks'),
    State('stepper', 'active'),
    State('session-store', 'data'),
    output=dict(
        pathname=Output('url_home', 'pathname')
    ),
    prevent_initial_call=True
)
def go_to_next_page(
    _,
    current_step,
    session_data
):
    if current_step < 4:
        raise PreventUpdate

    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
    embedding_id = session_data[StoreKey.EMBEDDING_SELECTION.value]

    if api.is_dataset_embedded(dataset_id, embedding_id):
        print("Home to annotation \n -------------------------- \n")
        pathname = f'/annotation/{dataset_id}'
    else:
        print("Home to embedding \n -------------------------- \n")
        pathname = f'/embedding'

    return dict(pathname=pathname)


clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='validateConfirmButton'),
    Output('confirm_button', "disabled"),
    Input("radio-selection", "value"),
)

# Alternative
# @callback(
#     Output(confirm_button, 'disabled'),
#     Input(RADIO_SELECTION, 'value'),
#     prevent_initial_call=True
# )
# def validate_confirm_button(radio_selection):
#     print("confirm_button_validation callback")
#     if radio_selection is None:
#         return True
#     else:
#         return False


# @callback(
#     Output({'type': 'step', 'index': MATCH}, 'loading'),
#     Output({'type': 'step', 'index': MATCH}, 'description'),
#     Input('stepper', 'active'),
#     State({'type': 'step', 'index': MATCH}, 'id'),
#     prevent_initial_call=True
# )
# def update_individual_step(active, step_id_dict):
#     step_id = step_id_dict["index"]
#     print("step_id", step_id)
#     print(type(step_id))
#
#     # step_id is a dictionary like {'type': 'step', 'index': <number>}
#     # You can also inspect callback_context if needed:
#     if active != step_id:
#         raise PreventUpdate
#
#     print("active matches step_id?")
#     return True, "IS this working"

