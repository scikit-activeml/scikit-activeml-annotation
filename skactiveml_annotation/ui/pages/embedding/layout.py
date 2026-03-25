
from dash import (
    Dash,
    register_page,
    dcc,
)

import dash_mantine_components as dmc

from . import (
    ids,
    callbacks,
)

def register(app: Dash):
    register_page(
        __name__,
        path_template='/embedding',
        layout=_layout,
        description='Embedding Page',
    )
    callbacks.register(app)


def _layout(**kwargs):
    _ = kwargs
    return (
        dmc.AppShellMain(
            [
                dcc.Location(ids.URL, refresh=True),
                dcc.Location(ids.INIT, refresh=False),
                dmc.Stack(
                    [
                        dmc.Title("Embedding", id=ids.TITLE),

                        dmc.Container(
                            id=ids.SELECTION_CONTAINER
                        ),

                        dmc.Group(
                            [
                                dmc.Button("Cancel", id=ids.CANCEL_BUTTON, disabled=True),
                                dmc.Button("Start Embedding", id=ids.CONFIRM_BUTTON)
                            ],
                            id=ids.EMBEDDING_BTN_CONTAINER
                        ),

                        dmc.Progress(
                            id=ids.EMBEDDING_PROGRESS,
                            value=0,
                            size="xl",
                            animated=True,
                            style={
                                'width': '75%',
                                'height': '30px',
                            },
                            transitionDuration=500
                        ),
                    ],
                    align='center'
                ),
            ]
        )
    )
