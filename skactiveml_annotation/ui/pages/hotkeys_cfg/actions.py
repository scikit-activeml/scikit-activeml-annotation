
from dash import (
    Dash,
    Input,
    State,
)

from skactiveml_annotation.shared_ids import KEYMAPPING_CFG
from skactiveml_annotation.ui import common
from skactiveml_annotation.ui.hotkeys import (
    ButtonAction,
    on_key_pressed_handler,
    register_action,
    register_default_keybinds,
)

from . import (
    ids,
)

RESET_ACTION = ButtonAction(
    "Hotkeys.Main.Reset",
    ids.RESET_HOTKEYS_BTN,
    "Reset",
)

CONFIRM_ACTION = ButtonAction(
    "Hotkeys.Main.Confirm",
    ids.CONFIRM_HOTKEYS_BTN,
    "Confirm",
)

BACK_ACTION = ButtonAction(
    "Hotkeys.Main.Back",
    ids.BACK_BTN,
    "Back",
)

ALL_ACTIONS = [
    RESET_ACTION,
    CONFIRM_ACTION,
    BACK_ACTION,
]


def register(app: Dash):
    for action in ALL_ACTIONS:
        register_action(action)

    register_default_keybinds(
        "Hotkeys",
        {
            "Main": {
                "Enter": CONFIRM_ACTION.action_id,
                "Backspace+Alt+Control": BACK_ACTION.action_id,
                "R+Alt+Control": RESET_ACTION.action_id,
            },
        }
    )

    @app.callback(
        Input(ids.KEYBOARD, "n_keydowns"),
        State(ids.KEYBOARD, "keydown"),
        State(KEYMAPPING_CFG, "data"),
        prevent_initial_call=True
    )
    def on_key_pressed(
        trigger,
        key_event,
        hotkey_cfg_json,
    ):
        hotkey_cfg = common.try_deserialize_hotkey_cfg(hotkey_cfg_json)
        on_key_pressed_handler(trigger, key_event, hotkey_cfg, 'Hotkeys')
    _ = on_key_pressed

