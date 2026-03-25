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

CONFIRM_ACTION = ButtonAction(
    "Home.Main.Confirm",
    ids.CONFIRM_BUTTON,
    "Confirm",
    "Confirm label selection for current sample and move on to the next sample"
)

BACK_ACTION = ButtonAction(
    "Home.Main.Back",
    ids.BACK_BUTTON,
    "Back",
    "Go back to previous sample"
)

ALL_ACTIONS = [
    CONFIRM_ACTION,
    BACK_ACTION,
]


def register(app: Dash):
    for action in ALL_ACTIONS:
        register_action(action)

    register_default_keybinds(
        "Home",
        {
            "Main": {
                "Enter": CONFIRM_ACTION.action_id,
                "Backspace+Alt+Control": BACK_ACTION.action_id,
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
        key_mappings_json,
    ):
        hotkey_cfg = common.try_deserialize_hotkey_cfg(key_mappings_json)
        on_key_pressed_handler(trigger, key_event, hotkey_cfg, 'Home')
    _ = on_key_pressed
