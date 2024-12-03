import dash
from dash import html

dash.register_page(__name__)


def layout(**kwargs):
    return (
        html.Div([
            html.H1('This is our Archive page'),
            html.Div('This is our Archive page content.'),
        ])
    )