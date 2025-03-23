
"""
AppShell with all elements

Navbar, header, aside and footer used together
"""
from dash import (
    dcc,
    Dash,
    callback,
    Input,
    Output,
    State,
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


def _create_dataset_selection() -> dmc.RadioGroup:
    dataset_options = get_dataset_config_options()
    data = [(cfg_name, f'{cfg.display_name} - ({instantiate(cfg.data_type).value})')
            for cfg_name, cfg in dataset_options.items()]

    # TODO repeated code.
    return (
        dmc.RadioGroup(
            # TODO needs to be unique across multiple pages?
            id='radio-selection',
            children=dmc.Stack([dmc.Radio(label=l, value=k) for k, l in data]),
            value=None,
            size="sm",
            style={'border': '2px solid red'}
        )
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
                                align='center'
                            ),
                            dmc.Flex(
                                [
                                    stepper := dmc.Stepper(
                                        [
                                            dmc.StepperStep(label="Dataset", description="Select a Dataset", loading=True),
                                            dmc.StepperStep(label="Adapter", description="Select an Adapter"),
                                            dmc.StepperStep(label="Query Strategy", description="Select a Query Strategy"),
                                            dmc.StepperStep(label="Model", description="Select a model (if required by query)"),
                                        ],
                                        id='stepper',
                                        active=0,
                                        orientation='vertical',
                                        iconSize=30,
                                        style={'border': '2px solid red'}
                                    ),
                                    selection_container := dmc.Container(
                                        _create_dataset_selection(),
                                        id='selection_container'
                                        # Current selection injected here
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


# TODO Can I split this off into 2 callbacks?
# Callback is doing too much.
@callback(
    Output(selection_container, 'children'),
    Output('session-store', 'data'),
    Output(stepper, 'active', allow_duplicate=True),
    Input(stepper, 'active'),
    State('session-store', 'data'),
    State('radio-selection', 'value'),
    prevent_initial_call=True,
)
def on_step_change(step, session_data, value):
    print(f'on_step_change callback with step={step}')

    print('selection value:', value)
    print()
    if session_data is None:
        session_data = {}

    data = None
    # TODO this can be simplified
    if step == 0:
        return _create_dataset_selection(), session_data, no_update
    if step == 1:
        # Dataset has been selected
        session_data[StoreKey.DATASET_SELECTION.value] = value

        adapter_options = get_adapter_config_options()
        data = [(cfg_name, cfg.display_name) for cfg_name, cfg in adapter_options.items()]
    if step == 2:
        # Adapter has been selected
        session_data[StoreKey.ADAPTER_SELECTION.value] = value

        qs_options = get_qs_config_options()
        data = [(cfg_name, cfg.display_name) for cfg_name, cfg in qs_options.items()]

    if step == 3:
        # Query Strategy has been selected

        # TODO qs_id
        # Rename query_id to
        query_id = value
        session_data[StoreKey.QUERY_SELECTION.value] = query_id

        # TODO write helper function for this.
        qs_cfg = get_query_cfg_from_id(query_id)

        if qs_cfg.model_agnostic:
            # Query strategy is model agnostic aka no model is needed.
            session_data[StoreKey.MODEL_SELECTION.value] = None
            return no_update, session_data, step + 1

        # Query strategy relies on a model
        model_options = get_model_config_options()
        data = [(cfg_name, cfg.display_name) for cfg_name, cfg in model_options.items()]

    if step == 4:
        session_data[StoreKey.MODEL_SELECTION.value] = value
        return no_update, session_data, no_update

    # TODO add helper method for that?

    children = dmc.RadioGroup(
        # TODO needs to be unique across multiple pages?
        id='radio-selection',
        children=dmc.Stack([dmc.Radio(label=l, value=k) for k, l in data]),
        value=None,
        size="sm",
        style={'border': '2px solid red'}
    )
    return children, session_data, no_update


@callback(
    Output(stepper, 'active'),
    Input(back_button, 'n_clicks'),
    Input(confirm_button, 'n_clicks'),
    State(stepper, 'active'),
    prevent_initial_call=True
)
def on_button_click(_, __, step):
    print("on_button_click callback")
    # Dash context to figure out which input triggered the callback.
    button_clicked = callback_context.triggered_id
    is_confirm_button = button_clicked == "confirm_button"

    if is_confirm_button:
        if step >= 4:
            raise PreventUpdate
        return step + 1
    else:
        return max(step - 1, 0)


@callback(
    Output(confirm_button, 'disabled'),
    Input('radio-selection', 'value'),
    prevent_initial_call=True
)
def validate_confirm_button(radio_selection):
    print("confirm_button_validation callback")
    if radio_selection is None:
        return True
    else:
        return False


@callback(
    Output(url_home, 'pathname'),
    Input(confirm_button, 'n_clicks'),
    State(stepper, 'active'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def go_to_annot_page(_, step, session_data):
    if step < 4:
        raise PreventUpdate

    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
    return f'/annotation/{dataset_id}'


# @callback(
#     Output({'type': 'stepper-step', 'index': MATCH}, 'description'),
#     Input({'type': 'stepper', 'index': MATCH}, 'active'),
#     State('radio-selection', 'label')
# )
# def update_step_description(step, selected_label):
#     return selected_label


# if __name__ == "__main__":
#     app.run(debug=True, port=9999)
