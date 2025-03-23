import logging

import dash
from dash import (
    Dash,
    Input,
    Output,
    html,
    dcc
)

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

from ui.components.navbar import create_navbar
from util.path import PAGES_PATH, DATA_CONFIG_PATH

from util.deserialize import parse_yaml_config_dir

app = Dash(
    __name__, 
    use_pages=True,  # Use dash page feature
    pages_folder=str(PAGES_PATH),
    external_stylesheets=[dbc.themes.BOOTSTRAP, dmc.theme.DEFAULT_THEME, dmc.styles.ALL],
    # Allows to register callbacks on components that will be created by other callbacks,
    # and are therefore not in the initial layout.
    suppress_callback_exceptions=True, 
    prevent_initial_callbacks=True, 
)

app.layout = (
    dmc.MantineProvider(
        [
            dcc.Store('session-store', storage_type='session'),
            create_navbar(),
            dmc.Container(
                dash.page_container,
                style={'border': '5px dashed red'}
            )
        ]
    )
)


if __name__ == "__main__":
    app.run(debug=True)
