from urllib.parse import urlencode

import dash
from dash import html, dcc, callback, Input, Output, State
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from hydra.utils import instantiate

from core.api import get_dataset_config_options, get_qs_config_options

dash.register_page(__name__, path='/')


# path variables and query string are captured from the URL and passed into kwargs
def layout(**kwargs):
    # TODO load other options aswell.
    dataset_options = get_dataset_config_options()
    qs_options = get_qs_config_options()

    return (
        dbc.Container(
            [
                dcc.Store(id='session-store-home'),
                dcc.Location(id='url-home', refresh=True),

                # Top Text
                dbc.Row(
                    dbc.Col(
                        [
                            html.H1("Welcome to scikit-activeml-annotation", className='text-center'),
                            html.H2('Configure your Pipeline', className='text-center'),
                        ],
                        width="auto",
                    ),
                    justify="center",
                    class_name='my-5',
                ),

                # Configure Section
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Accordion(
                                [

                                    # Dataset selection
                                    dbc.AccordionItem(
                                        [
                                            dcc.RadioItems(
                                                id="dataset-select",
                                                options=[
                                                    {
                                                        "label": f"{cfg.display_name} - ({instantiate(cfg.data_type).value})",
                                                        "value": f"{cfg_name}"
                                                    }
                                                    for cfg_name, cfg in dataset_options.items()
                                                ],
                                                value=None,  # Default selection
                                                className="form-check mx-1",
                                                # Adds Bootstrap form-check styling to the radio items
                                                inputStyle={'margin-right': '4px'}
                                            ),
                                        ],
                                        title="Select a Dataset",
                                        id='dataset-accordion-home'
                                    ),

                                    # Query Strategy selection
                                    dbc.AccordionItem(
                                        [
                                            dcc.RadioItems(
                                                id="qs-select",
                                                options=[
                                                    {"label": f"{cfg.name}", "value": f"{cfg_name}"}
                                                    for cfg_name, cfg in qs_options.items()
                                                ],
                                                value=None,  # Default selection
                                                className="form-check mx-1",
                                                # Adds Bootstrap form-check styling to the radio items
                                                inputStyle={'margin-right': '4px'}
                                            ),
                                        ],
                                        title="Select a Query Strategy",
                                        id='qs-accordion-home'
                                    )
                                ],
                                id='accordion-home',
                                active_item=False,
                                class_name='mb-3',
                                always_open=True,
                            ),

                            dbc.Button('Confirm Selection', n_clicks=0, id='select-button', color='dark',
                                       class_name='w-100', disabled=True),
                        ],
                        width=4
                    ),
                    justify='center',
                    class_name='my-20'
                )
            ],
            # fluid=True
        )
    )


@callback(
    Output('url-home', 'pathname'),
    # Output('url-home', 'search'),
    Input('select-button', 'n_clicks'),
    State('dataset-select', 'value'),
    prevent_initial_call=True
)
def on_button_confirm_home(n_clicks: int, value):
    if n_clicks == 0:
        return None, None,  # stay on the same page

    dataset_name = value
    print("[Home] Selected dataset: ", dataset_name)
    return f'/annotation/{dataset_name}'


# Validation
@callback(
    Output('select-button', 'disabled'),
    Input('dataset-select', 'value'),
    prevent_initial_call=True
)
def enable_button(value):
    if value is None:
        # No dataset was selected. Leave the button in the disabled state.
        return True  # , f'Dataset: {value}', []

    return False  # , f'Dataset: {value}', []


# Accordion
@callback(
    Output('dataset-accordion-home', 'title'),
    Output('accordion-home', 'active_item'),
    State('accordion-home', 'active_item'),
    Input('dataset-select', 'value'),
    prevent_initial_call=True
)
def collapse_accordion_item(active_item, value):
    print(active_item)
    print(value)
    return f'Dataset: {value}', []
