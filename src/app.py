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
)

app.layout = (
    html.Div(
        [
            create_navbar(),
            dbc.Container(
                dash.page_container,
                fluid=True
            )
        ]
    )
)


if __name__ == "__main__":
    app.run_server(debug=True)
