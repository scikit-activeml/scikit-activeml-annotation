from collections import Counter
import json
from dataclasses import dataclass
from typing import Final
import logging

import pydantic

from dash import (
    Dash,
    Output,
    Input,
    State,
    set_props,
)
from dash.exceptions import PreventUpdate

from skactiveml_annotation.shared_ids import CLICK_BTN_TRIGGER, KEYMAPPING_CFG
from skactiveml_annotation.ui.pages.home.ids import URL_INIT

MOD_KEY_MAPPING: Final = {"altKey": "Alt", "ctrlKey": "Control",
                          "shiftKey": "Shift", "metaKey": "Meta"}

VALID_SPECIAL_KEYS: Final = (
    "Unbound",
    "Enter",
    "Backspace",
    "CapsLock",
    "Tab",
    " ", # Space
    "Escape",
) + tuple(MOD_KEY_MAPPING.values())

@dataclass
class ButtonAction:
    action_id: str
    btn_id: str | dict[str, str]
    btn_text: str
    description: str = ""

# Lazy Initialization
__button_actions: dict[str, ButtonAction] = dict()
DEFAULT_KEYBINDS: dict[str, dict] = dict()


class HotkeyConfig(pydantic.BaseModel):
    mapping: dict = DEFAULT_KEYBINDS
    is_user_defined: bool = False


# --- API ---
def register_action(button_action: ButtonAction) -> ButtonAction:
    action_id = button_action.action_id

    if action_id in __button_actions:
        logging.warning(f"Action id: {action_id!r} is allready in use.")
    else:
        __button_actions[action_id] = button_action

    return button_action


def register_default_keybinds(page: str, page_bindings: dict[str, dict]) -> dict:
    DEFAULT_KEYBINDS[page] = page_bindings
    return page_bindings


def button_actions() -> dict[str, ButtonAction]:
    return __button_actions


def register_callbacks(app: Dash):
    @app.callback(
        Input(URL_INIT, "pathname"),
        State(KEYMAPPING_CFG, "data"),
        output=dict(
            hotkey_cfg=Output(KEYMAPPING_CFG, "data")
        )
    )
    def ensure_hotkeys_initialized(
        _,
        hotkey_cfg_json,
    ):
        if hotkey_cfg_json is None:
            return dict(
                hotkey_cfg=HotkeyConfig().model_dump()
            )

        try:
            hotkey_cfg = HotkeyConfig.model_validate(hotkey_cfg_json)
        except pydantic.ValidationError:
            logging.error(
                "Invalid hotkey configuration json; using defaults instead",
                exc_info=True
            )
            return dict(
                hotkey_cfg=HotkeyConfig().model_dump()
            )

        if hotkey_cfg.is_user_defined:
            # Dont override user defined hotkeys
            raise PreventUpdate

        # Updating non-user defined hotkeys to latest defaults
        hotkey_cfg = HotkeyConfig()

        return dict(
            hotkey_cfg=hotkey_cfg.model_dump()
        )
    _ = ensure_hotkeys_initialized


def on_key_pressed_handler(
    trigger,
    key_event,
    hotkey_cfg: HotkeyConfig,
    page: str,
    modal: str = "Main",
):
    # Prevent Key repeat events from doing anything
    if trigger is None or key_event["repeat"]:
        raise PreventUpdate

    mapping = hotkey_cfg.mapping

    modal_mapping = mapping.get(page, None)
    if modal_mapping is None:
        logging.error(f"Key Bindings for page: {page} does not exist")
        raise PreventUpdate

    key_mapping = modal_mapping.get(modal, None)
    if key_mapping is None:
        # Use Main page keybinds of keybinds for modal are not defined
        key_mapping = modal_mapping.get("Main", None)
    if key_mapping is None:
        logging.error(f"No Main key mapping is defined for page: {page}")
        raise PreventUpdate

    normalized_hotkey = _key_event_to_canonical_str(key_event)

    key_mapping = modal_mapping.get(modal, None)
    if key_mapping is None:
        logging.error(
            f"Key mapping for modal {modal!r} on page {page!r} does not exist"
        )
        raise PreventUpdate

    button_action_id = key_mapping.get(normalized_hotkey, None)
    if button_action_id is None:
        logging.debug(f"Key Combo: {normalized_hotkey} is not bound. No Action is fired.")
        raise PreventUpdate

    button_action = __button_actions[button_action_id]
    button_id = button_action.btn_id

    # Simulate a button click
    set_props(
        CLICK_BTN_TRIGGER,
        dict(data=button_id)
    )


def normalize_hotkey_str(key_combo: str) -> str:
    if key_combo == "":
        key_combo = "Unbound"

    parts = key_combo.split("+")
    parts = [part.strip().capitalize() for part in parts]
    first = parts[0]

    is_first_mod_key = first in MOD_KEY_MAPPING.values()
    if is_first_mod_key:
        start_at_idx = 0
    else:
        start_at_idx = 1

    # Only sort Mod keys
    parts = parts[:start_at_idx] + sorted([
        part for part in parts[start_at_idx:]
    ])

    key = parts[0]
    mod_keys = parts[1:]

    if len(key) > 1 and first not in VALID_SPECIAL_KEYS:
        raise ValueError(f"Key {key!r} is not a valid Key.")

    mod_counts = Counter(mod_keys)
    for mod, count, in mod_counts.items():
        # Check that not Mod key is specified twice
        if count > 1:
            raise ValueError(f"Modifier Key: {mod!r} is used multiple times")

        # Check only valid mod keys
        if mod not in MOD_KEY_MAPPING.values():
            raise ValueError(f"Modifer Key: {mod!r} is not valid")

    if len(parts) > 1:
        return key + "+" + "+".join(mod_keys)
    return key


# --- Helper Funcions ---
def _key_event_to_canonical_str(event: dict) -> str:
    # Allow to specify i.e. D instead of d
    key = event["key"].capitalize()

    if key in MOD_KEY_MAPPING.values():
        parts = []
    else:
        parts = [key]

    for mod_key in MOD_KEY_MAPPING:
        if event.get(mod_key, False):
            parts.append(MOD_KEY_MAPPING[mod_key])

    return "+".join(parts)
