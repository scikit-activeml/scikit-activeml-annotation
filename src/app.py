import logging

import dash
from dash import Dash, Input, Output
from dash import html, dcc

import dash_bootstrap_components as dbc

from ui.components.navbar import create_navbar
from util.path import PAGES_PATH, DATA_CONFIG_PATH

from util.deserialize import parse_yaml_config_dir

app = Dash(
    __name__, 
    use_pages=True,
    pages_folder=str(PAGES_PATH),
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # Allows to register callbacks on components that will be created by other callbacks,
    # and are therefore not in the initial layout.
    suppress_callback_exceptions=True, 
    prevent_initial_callbacks=True, 
)

app.layout = (
    dbc.Container(
        [
            create_navbar(),
            dbc.Container(
                dash.page_container,
                fluid=True,
                #style={'border': '5px dashed red',},
                # class_name='px-0'
            )
        ],
        fluid=True,
        style={'padding': 0}
    )
)


if __name__ == "__main__":
    app.run(debug=True)
