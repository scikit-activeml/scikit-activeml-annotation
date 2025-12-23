
import logging

from dash import (
    ALL,
    Output,
    Input, 
    State, 
    callback,
)

from skactiveml_annotation.ui.hotkeys import (
    ButtonAction,
    on_key_pressed_handler, 
    register_action,
    register_default_keybinds,
)
from . import ids

# ---------------------------
# Annotations Page Actions
# ---------------------------
CONFIRM = register_action(
    # INFO: Page.[Main, Modal].Action
    ButtonAction(
        "Annotation.Main.Confirm",
        {'type': 'action-button', 'index': 'confirm'},
        "Confirm",
        "Confirm label selection for current sample and move on to the next sample"
    ),
)

BACK = register_action(
    ButtonAction(
        "Annotation.Main.Back",
        {'type': 'action-button', 'index': 'back'},
        "Back",
        "Go back to previous sample"
    ),
)

DISCARD = register_action(
    ButtonAction(
        "Annotation.Main.Discard",
        {'type': 'action-button', 'index': 'discard'},
        "Discard",
        "Discard the current sample essentially marking it as an outlier"
    ),
)

SKIP = register_action(
    ButtonAction(
        "Annotation.Main.Skip",
        {'type': 'action-button', 'index': 'skip'},
        "Skip",
        "Skip the sample if you are unsure. This sample might be selected again."
    ),
)

APPLY = register_action(
    ButtonAction(
        "Annotation.Main.Apply",
        "apply-btn",
        "Apply",
    ),
)

OPEN_LABEL_SETTINGS = register_action(
    ButtonAction(
        "Annotation.Main.OpenLabelSettings",
        "label-setting-btn",
        "Open Label Settings Modal",
        ""
    ),
)

SKIP_BATCH = register_action(
    ButtonAction(
        "Annotation.Main.SkipBatch",
        "skip-batch-button",
        "Skip Batch",
        ""
    ),
)

# --- Modal Actions ---
CONFIRM_MODAL_ANNOTATION = register_action(
    ButtonAction(
        "Annotation.LabelSettingsModal.Confirm",
        "label-setting-confirm-btn",
        "Confirm Modal",
        "Confirm the modal"
    ),
)


#  --- Audio Controlls ---
# TODO:
# TOGGLE_AUDIO_PLAYBACK = register_default_keybinds(
#     ButtonAction(
#         "Annotation"."Main".
#
#     )
# )


# --- Default Keybinds ---
DEFAULT_KEYBINDS_ANNOTATION = register_default_keybinds(
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


@callback(
    Input("keyboard", "n_keydowns"),
    State("keyboard", "keydown"),
    State("keymapping-cfg", "data"),
    State({ 'type': 'modal', 'index': ALL}, "id"),
    State({ 'type': 'modal', 'index': ALL}, "opened"),
    prevent_initial_call=True
)
def on_annotation_key_pressed(
    trigger,
    key_event,
    key_mappings,
    modal_ids,
    modal_open_values,
):
    modal_id = "Main"
    for id, is_open in zip(modal_ids, modal_open_values):
        if is_open:
            modal_id = id['index']
            break

    on_key_pressed_handler(trigger, key_event, key_mappings, "Annotation", modal_id)
