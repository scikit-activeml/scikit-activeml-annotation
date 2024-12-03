import base64
from io import BytesIO

import dash
from dash import html, dcc, callback, Input, Output
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
from sklearn import datasets

from PIL import Image

from util.deserialize import parse_yaml_config_dir
from util.path import DATA_CONFIG_PATH

from core.service import SessionConfig, setup_activeMl_cycle

from urllib.parse import parse_qs

dash.register_page(__name__, path='/annotation')

def _build_side():
    pass

# Dummy labels for radio buttons
# LABELS = ['Label 1', 'Label 2', 'Label 3', 'Label 4']


def encode_base64(image: np.ndarray) -> str:
    # Normalize the image to 0–255 range and convert to uint8
    if image.max() > 1:  # Handle images already in range 0-255
        image = 255 * (image / image.max())
    image = image.astype(np.uint8)
    
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(image)
    # print(pil_image.size)
    pil_image = pil_image.resize((250, 250), Image.Resampling.NEAREST)
    
    # Save image to a BytesIO stream in PNG format
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the image as Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return the data URI
    return f"data:image/png;base64,{image_base64}"

# Triggers with each reload.
def load_image(idx: int) -> bytes:
    bunch = datasets.load_digits()
    labels = bunch.target
    label_names = bunch.target_names

    images = bunch.images
    image_url = encode_base64(images[idx])
    return image_url


def get_label_names() -> list[str]:
    bunch = datasets.load_digits()
    labels = bunch.target
    label_names = bunch.target_names
    return label_names

def create_sidebar():
    return dbc.Col(
        children=[
            dbc.Card(
                children=[
                    dbc.CardHeader("Settings"),
                    dbc.CardBody([
                        # Batch Size selection
                        html.Label('Batch Size'),
                        dcc.Input(
                            id='batch-size-input',
                            type='number',
                            value=10,
                            min=1,
                            step=1,
                            style={'width': '100%'}
                        ),
                        html.Br(),
                        html.Br(),
                        # Subsampling selection
                        html.Label('Subsampling'),
                        dcc.Input(
                            id='subsampling-input',
                            type='number',
                            value=1000,
                            min=1,
                            step=1,
                            style={'width': '100%'}
                        ),
                        html.Br(),
                        html.Br(),
                        # Cycles selection
                        html.Label('Cycles'),
                        dcc.Input(
                            id='cycles-input',
                            type='number',
                            value=10,
                            min=1,
                            step=1,
                            style={'width': '100%'}
                        ),
                    ])
                ]
            )
        ],
        style={'border': '2px solid red'},
        width=3
    )

def create_hero_section(label_idx: int):
    return dbc.Col(
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        html.Div(
                            children=[
                                html.Img(
                                    src=load_image(label_idx),
                                    # src="https://via.placeholder.com/600x400",
                                    style={'max-width': '100%', 'margin': '0 auto'}
                                ),
                            ],
                            style={'textAlign': 'center'}
                        ),
                        width=12
                    ),
                ],
                justify="center"
            ),
            dbc.Row(
                dbc.Col(
                    [
                        # Radio buttons for label selection, centered below the image
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    [
                                        html.Label('Select Label'),
                                        dcc.RadioItems(
                                            id='label-radio',
                                            options=[{'label': l_name, 'value': idx} for idx, l_name in enumerate(get_label_names())],
                                            value=0,  # Default to the first label
                                            labelStyle={'display': 'block', 'margin': '10px 0'}
                                        )
                                    ],
                                    style={'marginTop': '20px', 'border': '2px dashed brown'},
                                    width=6
                                ),
                            ],
                            justify="center",
                            style={'marginTop': '20px', 'border': '2px solid gold'}
                        ),
                        # Button to confirm selection
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    dbc.Button(
                                        'Confirm Selection',
                                        id='confirm-button',
                                        color='dark',
                                    ),
                                    width=12
                                )
                            ],
                            style={'marginTop': '20px'},
                            justify='center',
                        ),
                    ],
                    style={'border': '2px solid orange', 'textAlign': 'center'},
                ),
                style={
                'border': '2px green',
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center'},
                # style={'textAlign': 'center'},
                justify='center'
            ),

        ],
        width=9,
        style={'border': '2px solid blue'}
    )

def layout(index=None, **kwargs):
    if index == None:
        index = 0
    else:
        index = int(index)

    return html.Div(
        children=[
            dbc.Container(
                children=[
                    # Row containing the sidebar and the hero section
                    dbc.Row(
                        [   
                            dcc.Store(id='session-store', storage_type='session'),
                            dcc.Location(id='url-annotation', refresh=True),
                            create_sidebar(),  # Sidebar with settings
                            create_hero_section(index),  # Main content area (image, labels, button)
                        ],
                        style={'border': '2px solid red'}
                    )
                ],
                style={'border': '2px solid green'},
                fluid=True  # Use fluid layout for full width
            )
        ]
    )

def get_query_value(query_str: str, key: str) -> str:
    query_params = parse_qs(query_str.lstrip('?'))
    if not key in query_params:
        return None
    return query_params[key][0]

def build_query(query: dict[str, str] | tuple[str, str]) -> str:
    if isinstance(query, tuple):
        key, value = query
        return f'?{key}={value}'
    return NotImplementedError


@callback(
    Output('url-annotation', 'pathname'),
    Output('session-store', 'data'),
    Input('confirm-button', 'n_clicks'),
    State('label-radio', 'value'),
    State('url-annotation', 'search'),
    State('session-store', 'data')
)
def on_button_click(n_clicks: int, value: int, search: str, data: dict | None):
    session_cfg = SessionConfig()
    cycle = setup_activeMl_cycle(session_cfg)

    gen = cycle()
    query_idx = next(gen)
    print(query_idx)
    
    """ query_idx = cycle.send(None)
    print(query_idx) """

    print(get_query_value(search, 'index'))
    
    if data is None:
        print("Datastore is not initialized!")

    # Initially n_clicks is None?
    if n_clicks is None:
        return None, None
    
    """ data = {
        'initialized': True,
    }

    print(f'Selected Label Name is: {value}')

    return {
        'data': data
    } """

    return None, None
    
