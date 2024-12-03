import dash_bootstrap_components as dbc
import dash
from dash import Input, Output, State, html, callback
from dash_bootstrap_components._components.Container import Container

def create_navbar(**kwargs):
    return (
        dbc.Navbar(
            dbc.Container(
                [
                    # Navbar links (aligned to the left)
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.Div(page["name"], className="ms-2"),
                                    href=page["path"],
                                    active="exact",
                                )
                            )
                            for page in dash.page_registry.values()
                        ],
                        pills=True,
                        class_name="ml-0"  # Align nav items to the left
                    )
                ]
            ),
        color="dark",
        dark=True,         
        ) 
    )



