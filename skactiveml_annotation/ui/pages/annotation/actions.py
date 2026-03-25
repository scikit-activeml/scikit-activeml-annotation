from dash import (
    ALL,
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

CONFIRM = ButtonAction(
    # Page.[Main, Modal].Action
    "Annotation.Main.Confirm",
    {'type': ids.ACTION_BTN, 'index': 'confirm'},
    "Confirm",
    "Confirm label selection for current sample and move on to the next sample"
)


BACK = ButtonAction(
    "Annotation.Main.Back",
    {'type': ids.ACTION_BTN, 'index': 'back'},
    "Back",
    "Go back to previous sample"
)

DISCARD = ButtonAction(
    "Annotation.Main.Discard",
    {'type': ids.ACTION_BTN, 'index': 'discard'},
    "Discard",
    "Discard the current sample essentially marking it as an outlier"
)

SKIP = ButtonAction(
    "Annotation.Main.Skip",
    {'type': ids.ACTION_BTN, 'index': 'skip'},
    "Skip",
    "Skip the sample if you are unsure. This sample might be selected again."
)

APPLY = ButtonAction(
    "Annotation.Main.Apply",
    ids.APPLY_PRESENTION_SETTINGS_BTN,
    "Apply",
)

OPEN_LABEL_SETTINGS = ButtonAction(
    "Annotation.Main.OpenLabelSettings",
    ids.LABEL_SETTING_BTN,
    "Open Label Settings Modal",
)

SKIP_BATCH = ButtonAction(
    "Annotation.Main.SkipBatch",
    ids.SKIP_BATCH_BTN,
    "Skip Batch",
)

# --- Modal Actions ---
CONFIRM_MODAL_ANNOTATION = ButtonAction(
    "Annotation.LabelSettingsModal.Confirm",
    ids.LABEL_SETTING_CONFIRM_BTN,
    "Confirm Modal",
    "Confirm the modal"
)


ALL_ACTIONS = [
    CONFIRM,
    BACK,
    DISCARD,
    SKIP,
    APPLY,
    OPEN_LABEL_SETTINGS,
    SKIP_BATCH,
    CONFIRM_MODAL_ANNOTATION,
]


def register(app: Dash):
    for action in ALL_ACTIONS:
        register_action(action)

    register_default_keybinds(
        "Annotation",
        {
            "Main": {
                "Enter": CONFIRM.action_id,
                "Backspace+Alt+Control": BACK.action_id,
                "D+Alt+Control": DISCARD.action_id,
                "S+Alt+Control": SKIP.action_id,
                "L+Alt+Control": OPEN_LABEL_SETTINGS.action_id,
                "B+Alt+Control": SKIP_BATCH.action_id,
                "A+Alt+Control": APPLY.action_id,
            },
            "LabelSettingsModal": {
                "Enter": CONFIRM_MODAL_ANNOTATION.action_id,
            },
        }
    )

    @app.callback(
        Input(ids.KEYBOARD, "n_keydowns"),
        State(ids.KEYBOARD, "keydown"),
        State(KEYMAPPING_CFG, "data"),
        State({ 'type': 'modal', 'index': ALL}, "id"),
        State({ 'type': 'modal', 'index': ALL}, "opened"),
        prevent_initial_call=True
    )
    def on_key_pressed(
        trigger,
        key_event,
        key_mappings_json,
        modal_ids,
        modal_open_values,
    ):
        modal_id = "Main"
        for id, is_open in zip(modal_ids, modal_open_values):
            if is_open:
                modal_id = id['index']
                break

        hotkey_cfg = common.try_deserialize_hotkey_cfg(key_mappings_json)
        on_key_pressed_handler(trigger, key_event, hotkey_cfg, "Annotation", modal_id)
    _ = on_key_pressed
