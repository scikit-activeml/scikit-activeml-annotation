import dash
from dash import html, dcc, callback, Input, Output

import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output

from util.deserialize import parse_yaml_config_dir
from util.path import DATA_CONFIG_PATH

# Dummy data for dropdown
dropdown_options = [
    {'label': 'Option 1', 'value': 'opt1'},
    {'label': 'Option 2', 'value': 'opt2'},
    {'label': 'Option 3', 'value': 'opt3'}
]

dash.register_page(__name__, path='/')

def load_dataset_options() -> list[str]:
    out = []
    config_options = parse_yaml_config_dir(DATA_CONFIG_PATH)
    for config in config_options:
        out.append(config.name)
    return out


# path variables and query string are captured from the URL and passed into kwargs!
def layout(**kwargs):
    return (
        dbc.Container([
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
                                            for dataset_name in load_dataset_options()
                                        ],
                                        value="item_1",  # Default selection
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
                    dbc.Button('Select', n_clicks=0, id='select-button', color='dark', class_name='w-100'),
                ], 
                width="auto")  # "auto" for auto-width and centering the column
            , 
            justify="center")  # Centers the column (hence the button)
        ]),
    )

@callback(
    Output('url-home', 'pathname'),
    Input('select-button', 'n_clicks'),
    prevent_initial_call=True
)
def on_button_click(n_clicks):
    if n_clicks:
        return "/annotation"
    
    return None # stay on the same page