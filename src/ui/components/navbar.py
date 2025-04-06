import dash
from dash import html

import dash_mantine_components as dmc


def create_navbar(**kwargs):
    return dmc.AppShellHeader(
        children=[
            dmc.Group(
                children=[
                    dmc.Anchor(
                        children=page["name"],
                        href=page["path"],
                        # underline=False,
                        size="md",
                        style={"marginRight": "1rem"}
                    )
                    for page in dash.page_registry.values()
                ],
                align="center"
            )
        ],
        # width={"base": 250},  # Adjust the width as needed
        # p="md",
        # withBorder=True
    )



