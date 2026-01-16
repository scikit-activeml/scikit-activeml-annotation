from collections import defaultdict
from typing import Final

import dash
from dash import (
    ALL,
    dcc,
    callback,
    Input,
    Output,
    State,
    register_page,
)
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from dash_extensions import Keyboard

from skactiveml_annotation.ui import common
from skactiveml_annotation.ui.hotkeys import (
    ButtonAction,
    HotkeyConfig, 
    button_actions, 
    normalize_hotkey_str,
    on_key_pressed_handler,
    register_action,
    register_default_keybinds
)
from skactiveml_annotation.util import logging


register_page(
    __name__, 
    path='/hotkeys',
)


RESET_ACTION = register_action(
    ButtonAction(
        "Hotkeys.Main.Reset",
        "reset-hotkeys-btn",
        "Reset",
    ),
)

CONFIRM_ACTION = register_action(
    ButtonAction(
        "Hotkeys.Main.Confirm",
        "confirm-hotkeys-btn",
        "Confirm",
    ),
)


BACK_ACTION = register_action(
    ButtonAction(
        "Hotkeys.Main.Back",
        "back-hotkeys-btn",
        "Back",
    ),
)


DEFAULT_KEYBINDS_ANNOTATION = register_default_keybinds(
    "Hotkeys",
    {
        "Main": {
            "Enter": CONFIRM_ACTION.action_id,
            "Backspace+Alt+Control": BACK_ACTION.action_id,
            "R+Alt+Control": RESET_ACTION.action_id,
        },
    }
)


@callback(
    Input("hotkeys-keyboard", "n_keydowns"),
    State("hotkeys-keyboard", "keydown"),
    State("keymapping-cfg", "data"),
    prevent_initial_call=True
)
def on_home_key_pressed(
    trigger,
    key_event,
    hotkey_cfg_json,
):
    hotkey_cfg = common.try_deserialize_hotkey_cfg(hotkey_cfg_json)
    on_key_pressed_handler(trigger, key_event, hotkey_cfg, 'Hotkeys')


def layout(**kwargs: object):
    _ = kwargs

    return (
        dmc.Center(
            [
                dcc.Location(id='url-hotkeys', refresh=True),
                dcc.Location(id='url-hotkeys-init', refresh=False),

                dcc.Store(id="hotkey-ui-trigger"),

                Keyboard(
                    id="hotkeys-keyboard",
                ),

                dmc.Stack(
                    [
                        dmc.ScrollArea(
                            dmc.Container(
                                id="hotkey-configuration-container",
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
                                dmc.Button("Reset", id="reset-hotkeys-btn", color="dark"),
                                dmc.Button("Confirm", id="confirm-hotkeys-btn", color="dark"),
                                dmc.Button("Back", id="back-hotkeys-btn", color="dark"),
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


# TODO:
def _camel_case_to_title(s: str) -> str:
    # Split before capital letters that are followed by lowercase (normal word start)
    # or when a lowercase is followed by a capital (e.g., "HTMLParser")
    import re
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', s)
    return " ".join(parts)

@callback(
    Input("hotkey-ui-trigger", "data"),
    State("keymapping-cfg", "data"),
    output=dict(
        hotkey_cfg_container=Output("hotkey-configuration-container", "children")
    ),
    prevent_initial_call=True
)
def update_hotkey_page(
    _,
    hotkey_cfg_json,
):
    mapping = HotkeyConfig.model_validate(hotkey_cfg_json).mapping
    content = dmc.Stack(
        [
            dmc.Stack(
                [
                    dmc.Text(f"{page} Page", size="lg"),

                    dmc.Stack(
                        [
                            dmc.Stack(
                                [
                                    dmc.Text(_camel_case_to_title(modal), size="md") if modal != "Main" else None,
                                    
                                    dmc.Stack(
                                        [
                                            dmc.Flex(
                                                [
                                                    dmc.TextInput(
                                                        value=key_combo,
                                                        id={"type": "hotkey-input", "action": action_id},
                                                    ),

                                                    dmc.Box(button_actions()[action_id].btn_text),
                                                ],
                                                gap="md",
                                                align="center",
                                            )
                                            # action_id: i.e. Home.Main.Confirm
                                            # for modal, key_mapping in modal_mapping.items()
                                            for key_combo, action_id in key_mapping.items()
                                        ],
                                    )
                                ]
                            )
                            for modal, key_mapping in modal_mapping.items()
                        ]
                    ),

                ],
            )
            for page, modal_mapping in mapping.items()
        ],
    )

    return dict(
        hotkey_cfg_container=content
    )


@callback(
    Input('url-hotkeys-init', "pathname"),
    output=dict(
        ui_trigger=Output("hotkey-ui-trigger", "data")
    )
)
def init_hotkey_page(
    _,
):
    return dict(
        ui_trigger=dict(data=True)
    )


@callback(
    Input('confirm-hotkeys-btn', "n_clicks"),
    State({"type": "hotkey-input", "action": ALL}, "id"),
    State({"type": "hotkey-input", "action": ALL}, "value"),
    output=dict(
        hotkey_cfg=Output("keymapping-cfg", "data", allow_duplicate=True),
        ui_trigger=Output("hotkey-ui-trigger", "data", allow_duplicate=True),
        errors=Output({"type": "hotkey-input", "action": ALL}, "error"),
    ),
    prevent_initial_call=True
)
def on_hotkey_cfg_change_confirmed(
    n_clicks: int | None,
    ids,
    updated_hotkeys
):
    if n_clicks is None:
        raise PreventUpdate

    num_inputs: Final = len(ids)

    errors: list[str | bool] = [False] * num_inputs

    # Create a 2-level nested defaultdict:
    # updated_cfg[key1][key2] is always initialized to an empty dict.
    # See: https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict
    updated_cfg = defaultdict(lambda: defaultdict(dict))

    for idx, (entry_id, new_hotkey) in enumerate(zip(ids, updated_hotkeys)):
        action_id = entry_id['action']

        # i.e. Page.Modal.Confirm
        page, modal, action = action_id.split(".")
        logging.debug15("page, modal, action:")
        logging.debug15(page, modal, action)

        try:
            new_normalized_hotkey = normalize_hotkey_str(new_hotkey)

            if new_normalized_hotkey in updated_cfg[page][modal]:
                errors[idx] = f"Hotkey: {new_normalized_hotkey!r} is bound multiple times."

            updated_cfg[page][modal][new_normalized_hotkey] = action_id

        except ValueError as e:
            errors[idx] = str(e)


    if any(errors):
        return dict(
            hotkey_cfg=dash.no_update,
            ui_trigger=dash.no_update,
            errors=errors,
        )

    logging.debug15("updated_cfg:\n", updated_cfg)

    return dict(
        hotkey_cfg=HotkeyConfig(
            mapping=updated_cfg,
            is_user_defined=True,
        ).model_dump(),
        ui_trigger=dict(data=True),
        errors=[False] * num_inputs,
    )


@callback(
    Input('reset-hotkeys-btn', "n_clicks"),
    output=dict(
        hotkey_cfg=Output("keymapping-cfg", "data", allow_duplicate=True),
        ui_trigger=Output("hotkey-ui-trigger", "data", allow_duplicate=True)
    ),
    prevent_initial_call=True
)
def reset_hotkeys_to_default(
    n_clicks: int | None,
):
    if n_clicks is None:
        raise PreventUpdate

    return dict(
        hotkey_cfg=HotkeyConfig().model_dump(),
        ui_trigger=dict(data=True)
    )
