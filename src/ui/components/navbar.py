import dash
from dash import html
import dash_bootstrap_components as dbc

def create_navbar(**kwargs):
    return (
        dbc.Navbar(
            dbc.Container(
                [
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
            class_name='sticky-top'
        ) 
    )



