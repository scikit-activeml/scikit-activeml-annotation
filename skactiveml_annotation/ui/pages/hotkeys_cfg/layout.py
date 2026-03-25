
from dash import (
    Dash,
    dcc,
    register_page,
)

from dash_extensions import Keyboard

import dash_mantine_components as dmc

from . import (
    ids,
    actions,
    callbacks,
)


def register(app: Dash):
    register_page(
        __name__, path='/hotkeys', layout=_layout,
    )

    actions.register(app)
    callbacks.register(app)


def _layout(**kwargs: object):
    _ = kwargs
    return (
        dmc.Center(
            [
                dcc.Location(id=ids.URL, refresh=True),
                dcc.Location(id=ids.URL_INIT, refresh=False),

                dcc.Store(id=ids.UI_UPDATE_TRIGGER),

                Keyboard(
                    id=ids.KEYBOARD,
                ),

                dmc.Stack(
                    [
                        dmc.ScrollArea(
                            dmc.Container(
                                id=ids.CONFIGURATION_CONTAINER,
                                py="xs",
                            ),
                            type='auto',
                            offsetScrollbars='y',
                            styles=dict(
                                viewport={
                                    'maxHeight': '85vh',
                                    # 'border': '5px dashed red',
                                },
                            ),
                            py='xs',
                        ),
                        dmc.Flex(
                            [
                                dmc.Button("Reset", id=ids.RESET_HOTKEYS_BTN, color="dark"),
                                dmc.Button("Confirm", id=ids.CONFIRM_HOTKEYS_BTN, color="dark"),
                                dmc.Button("Back", id=ids.BACK_BTN, color="dark"),
                            ],
                            gap="md",
                        )
                    ],
                    gap="xl",
                )
            ],
            style={
                # "height": "100vh",
                # "border": "2px dashed green",
            }
        )
    )
