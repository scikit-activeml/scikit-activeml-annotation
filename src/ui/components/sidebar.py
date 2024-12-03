import dash
from dash import html
import dash_bootstrap_components as dbc


def create_sidebar(**kwargs):
    return (
        html.Div([
            dbc.Nav(
                [
                    dbc.NavLink(
                        html.Div(page["name"], className="ms-2"),
                        href=page["path"],
                        active="exact",
                    )
                    for page in dash.page_registry.values() # if page["path"].startswith("/topic")
                ],
                vertical=True,
                pills=True,
                class_name="bg-light",
            )
        ])
    )

