from urllib.parse import urlencode

import dash
from dash import html, dcc, callback, Input, Output, State
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from core.api import load_dataset_options

dash.register_page(__name__, path='/')

# path variables and query string are captured from the URL and passed into kwargs!
def layout(**kwargs):
    # TODO load other options aswell.
    dataset_options = load_dataset_options()
    return (
        dbc.Container([
            dcc.Store(id='session-store-home'),
            dcc.Location(id='url-home', refresh=True),
            dbc.Row(
                dbc.Col([
                    html.H1("Welcome to scikit-activeml-annotation", className='text-center'),
                    html.H2('Select a Dataset', className='text-center'),
                    
                    dbc.Accordion(
                        [   
                            dbc.AccordionItem(
                                [
                                    dcc.RadioItems(
                                        id="dataset-select",
                                        options=[
                                            {"label": f"{dataset_name}", "value": f"{dataset_name}"}
                                            for dataset_name in dataset_options
                                        ],
                                        value=None,  # Default selection
                                        labelStyle={"display": "block"},  # Makes each option appear like a list
                                        className="form-check",  # Adds Bootstrap form-check styling to the radio items
                                    ),
                                ],
                                title="Datasets",
                            ),
                        ],
                        active_item=False,
                        class_name='mb-5'
                    ),
                    dbc.Button('Select', n_clicks=0, id='select-button', color='dark', class_name='w-100', disabled=True),
                ], 
                width="auto")  # "auto" for auto-width and centering the column
            , 
            justify="center")  # Centers the column (hence the button)
        ]),
    )

@callback(
    # Output('session-store-home', 'data'),
    Output('url-home', 'pathname'),
    # Output('url-home', 'search'),
    Input('select-button', 'n_clicks'),
    State('dataset-select', 'value'),
    # State('session-store-home', 'data'),
    prevent_initial_call=True
)
def on_button_confirm_home(n_clicks: int, value):
    if n_clicks == 0:
        return None, None, # stay on the same page

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
        return True
    
    return False
    

