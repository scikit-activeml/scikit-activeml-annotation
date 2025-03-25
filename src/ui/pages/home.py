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

from hydra.utils import instantiate

from core.api import (
    get_dataset_config_options,
    get_qs_config_options,
    get_model_config_options,
    get_adapter_config_options
)

# TODO need to use some cache instead of session store for certain things.

from ui.storekey import StoreKey
from core.api import get_query_cfg_from_id

register_page(__name__, path='/')
app = Dash(
    __name__,
    prevent_initial_callbacks=True,
    # suppress_callback_exceptions=True
)


def _create_dataset_selection(preselect):
    print("_create data selection invoked")
    dataset_options = get_dataset_config_options()
    data = [(cfg.id, f'{cfg.display_name} - ({instantiate(cfg.data_type).value})')
            for cfg in dataset_options]

    # TODO repeated code.
    return (
        dmc.RadioGroup(
            # TODO needs to be unique across multiple pages?
            id='radio-selection',
            children=dmc.Stack([dmc.Radio(label=l, value=k) for k, l in data]),
            value=preselect,
            size="sm",
            style={'border': '2px solid red'}
        )
    )


# Helper function to build UI for different steps
def build_step_ui(step, session_data):
    if step == 0:
        return _create_dataset_selection(session_data.get(StoreKey.DATASET_SELECTION.value))
    elif step == 1:
        return create_radio_group(get_adapter_config_options(), session_data.get(StoreKey.ADAPTER_SELECTION.value))
    elif step == 2:
        return create_radio_group(get_qs_config_options(), session_data.get(StoreKey.QUERY_SELECTION.value))
    elif step == 3:
        return create_radio_group(get_model_config_options(), session_data.get(StoreKey.MODEL_SELECTION.value))
    return None


# Helper function to create a radio group
def create_radio_group(options, preselect):
    formatted_options = [(cfg.id, cfg.display_name) for cfg in options]
    return dmc.RadioGroup(
        id='radio-selection',
        children=dmc.Stack([dmc.Radio(label=l, value=k) for k, l in formatted_options]),
        value=preselect,
        size="sm",
        style={'border': '2px solid red'},
    )


# region Layout
stepper = dmc.Stepper(
    [
        dmc.StepperStep(id={'type': 'step', 'index': 0}, label="Dataset", description="Select a Dataset"),  # loading=True),
        dmc.StepperStep(id={'type': 'step', 'index': 1}, label="Adapter", description="Select an Adapter"),
        dmc.StepperStep(id={'type': 'step', 'index': 2}, label="Query Strategy", description="Select a Query Strategy"),
        dmc.StepperStep(id={'type': 'step', 'index': 3}, label="Model", description="Select a model (if required by query)"),
    ],
    id='stepper',
    active=0,
    # TODO consider using horizontal orientation
    orientation='vertical',
    iconSize=30,
    style={'border': '2px solid red'}
)


layout = dmc.AppShell(
    [
        dmc.AppShellMain(
            dmc.Center(
                [
                    url_home := dcc.Location(id='url_home', refresh=True),
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
                                    stepper,
                                    dcc.Loading(
                                        selection_container := dmc.Container(
                                            # Current selection injected here
                                            _create_dataset_selection(None),
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
                                    back_button := dmc.Button("Back", id='back_button'),
                                    confirm_button := dmc.Button("Confirm", id='confirm_button', disabled=True)
                                ]
                            )
                        ],
                        align='center',
                        style={'border': '2px solid gold'}
                    )
                ],
                style={'height': '100%'}
            )
        ),
        # dmc.AppShellAside("Aside", p="md"),
        # dmc.AppShellFooter("Footer", p="md"),
    ],
    padding="md",
    id="appshell",
)
# endregion


@callback(
    Input(confirm_button, 'n_clicks'),
    State('radio-selection', 'value'),
    State(stepper, 'active'),
    State('session-store', 'data'),
    output=dict(
        children=Output(selection_container, 'children', allow_duplicate=True),
        session_data=Output('session-store', 'data'),
        active=Output(stepper, 'active', allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def handle_confirm(
    _,
    radio_value,
    current_step,
    session_data
):
    print(f"handle_confirm triggered at step {current_step} with radio_value: {radio_value}")

    if current_step >= 4 or radio_value is None:
        raise PreventUpdate

    if session_data is None:
        session_data = {}

    new_step = current_step

    if current_step == 0:
        session_data[StoreKey.DATASET_SELECTION.value] = radio_value
        new_step = 1

    elif current_step == 1:
        session_data[StoreKey.ADAPTER_SELECTION.value] = radio_value
        new_step = 2

    elif current_step == 2:
        session_data[StoreKey.QUERY_SELECTION.value] = radio_value
        qs_cfg = get_query_cfg_from_id(radio_value)
        if qs_cfg.model_agnostic:
            session_data[StoreKey.MODEL_SELECTION.value] = None
            new_step = 4  # Skip model selection if model agnostic
        else:
            new_step = 3

    elif current_step == 3:
        session_data[StoreKey.MODEL_SELECTION.value] = radio_value
        new_step = 4

    return dict(
        children=build_step_ui(new_step, session_data),
        session_data=session_data,
        active=new_step
    )


# Back button callback
@callback(
    Input(back_button, 'n_clicks'),
    State(stepper, 'active'),
    State('session-store', 'data'),
    output=dict(
        children=Output(selection_container, 'children'),
        active=Output(stepper, 'active')
    ),
    prevent_initial_call=True
)
def handle_back(
    _,
    current_step,
    session_data
):
    print("handle_back callback")
    next_step = max(current_step - 1, 0)
    if current_step == 0 and next_step == 0:
        raise PreventUpdate

    return dict(
        children=build_step_ui(next_step, session_data),
        active=next_step
    )


@callback(
    Input(confirm_button, 'n_clicks'),
    State(stepper, 'active'),
    State('session-store', 'data'),
    output=dict(
        pathname=Output(url_home, 'pathname')
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
    Output(confirm_button, "disabled"),
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

