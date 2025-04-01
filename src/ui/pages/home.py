from dash import (
    dcc,
    Dash,
    callback,
    clientside_callback,
    ClientsideFunction,
    Input,
    Output,
    State,
    ALL,
    html,
    MATCH,
    Patch,
    no_update,
    register_page,
    callback_context,
)
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc
from dash_iconify import DashIconify

from hydra.utils import instantiate

from core.api import (
    get_dataset_config_options,
    get_qs_config_options,
    get_model_config_options,
    get_adapter_config_options,
    is_dataset_embedded,
    dataset_path_exits
)

from ui.storekey import StoreKey
from core.api import get_query_cfg_from_id

register_page(__name__, path='/')
app = Dash(
    __name__,
    prevent_initial_callbacks=True,
    # suppress_callback_exceptions=True
)


# region Layout
def layout():
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
                                        html.Div(id='radio-selection'),  # workaround so id exists at the start
                                        id='selection_container',
                                        # TODO use Mantine styling for this.
                                        style={"width": "15vw", "whiteSpace": "normal", "wordWrap": "break-word"}
                                    ),
                                    type='circle',
                                )
                            ]
                        ),
                        dmc.Group(
                            [
                                dmc.Button("Back", id='back_button'),
                                dmc.Button("Confirm", id='confirm_button', disabled=True)
                            ]
                        )
                    ],
                    align='center',
                    style={'border': '2px solid gold'}
                )
            ],
            style={'height': '100%'}
        )
    )


# Helper function to build UI for different steps
def create_step_ui(step, session_data):
    if step == 0:
        if session_data is None:
            preselect = None
        else:
            preselect = session_data.get(StoreKey.DATASET_SELECTION.value)
        return _create_dataset_selection(preselect)
    elif step == 1:
        return _create_adapter_radio_group(session_data)
    elif step == 2:
        return _create_radio_group(get_qs_config_options(), session_data.get(StoreKey.QUERY_SELECTION.value))
    elif step == 3:
        return _create_radio_group(get_model_config_options(), session_data.get(StoreKey.MODEL_SELECTION.value))
    return None


def create_stepper():
    return (
        dmc.Stepper(
            [
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 0}, label="Dataset", description="Select a Dataset"),  # loading=True),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 1}, label="Embedding", description="Select embedding method"),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 2}, label="Query Strategy", description="Select a Query Strategy"),
                dmc.StepperStep(id={'type': 'stepper-step', 'index': 3}, label="Model", description="Select a model"),
            ],
            id='stepper',
            active=0,
            # TODO consider using horizontal orientation
            orientation='vertical',
            iconSize=30,
            style={'border': '2px solid red'},
            allowNextStepsSelect=False
        )
    )


def _create_dataset_selection(preselect):
    print("_create data selection invoked")
    dataset_options = get_dataset_config_options()
    data = [(cfg, f'{cfg.display_name} - ({instantiate(cfg.data_type).value})')
            for cfg in dataset_options]

    # TODO repeated code.
    return (
        dmc.RadioGroup(
            id='radio-selection',
            children=dmc.Stack(
                [dmc.Radio(
                    label=cfg_display,
                    value=cfg.id,
                    disabled=dataset_path_exits(cfg.data_path)
                    )
                 for cfg, cfg_display in data]
            ),
            value=preselect,
            size="sm",
            style={'border': '2px solid red'}
        )
    )


def _create_adapter_radio_group(session_data):
    options = get_adapter_config_options()
    formatted_options = [(cfg.id, cfg.display_name) for cfg in options]

    preselect = session_data.get(StoreKey.ADAPTER_SELECTION.value)

    return dmc.RadioGroup(
        id='radio-selection',
        children=dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.Radio(label=cfg_name, value=cfg_id),
                        _create_bool_icon(is_dataset_embedded(
                            session_data[StoreKey.DATASET_SELECTION.value],
                            cfg_id
                        ))
                    ]
                )
                for cfg_id, cfg_name in formatted_options
            ]
        ),
        value=preselect,
        size="sm",
        style={'border': '2px solid red'},
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
        id='radio-selection',
        children=dmc.Stack([dmc.Radio(label=l, value=k) for k, l in formatted_options]),
        value=preselect,
        size="sm",
        style={'border': '2px solid red'},
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
    State('radio-selection', 'value'),
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
    # TODO initialize session_data somewhere else.
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
        session_data[StoreKey.ADAPTER_SELECTION.value] = radio_value

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
def go_to_annot_page(
    _,
    current_step,
    session_data
):
    print("go_to_annotation_page callback")
    if current_step < 4:
        raise PreventUpdate

    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
    return dict(pathname=f'/annotation/{dataset_id}')


clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='validateConfirmButton'),
    Output('confirm_button', "disabled"),
    Input("radio-selection", "value"),
)

# Alternative
# @callback(
#     Output(confirm_button, 'disabled'),
#     Input('radio-selection', 'value'),
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

