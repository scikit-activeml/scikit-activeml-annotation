import dash
from dash import html

import dash_bootstrap_components as dbc

from ..components.sidebar import create_sidebar

dash.register_page(__name__, name="Topics")


def layout(**kwargs):
    return (
        dbc.Row([
            dbc.Col(create_sidebar(), width=2), dbc.Col(html.Div("Topics Home Page"), width=10)
        ])
    )