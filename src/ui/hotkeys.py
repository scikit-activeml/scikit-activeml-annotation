import json

from dash import (
    html,
    register_page,
    callback,
    Input,
    Output,
    State,
    callback_context,
    clientside_callback,
    ClientsideFunction,
    set_props,
    ALL
)

from dash.exceptions import PreventUpdate

from ui.pages.annotation.ids import *

from dash import dcc
from dash_extensions import Keyboard

CAPTURE_KEY_TRIGGER = 'capture-key-trigger'

HOTKEY_CFG_ID = 'hotkey_cfg'

# Hotkey -> Keyboard -> on_capture_key callback ->
KEYBOARD_ID = 'keyboard'

# annotation_hotkeys = {
#     'Enter': LABEL_CONFIRM_BUTTON,
#     'B+shift': LABEL_BACK_BUTTON,
#     'D+shift': LABEL_DISCARD_BUTTON,
#     'S+shift': LABEL_SKIP_BUTTON,
#     'S+shift+alt': SKIP_BATCH_BUTTON,
# }
#
# home_hotkeys = {
#     'Enter': 'confirm_button',
#     'B+shift': 'back_button',
# }

hotkeys = {
    '': {
        'Enter': 'confirm_button',
        'B+shift': 'back_button',
    },
    'annotation': {
        'Enter': LABEL_CONFIRM_BUTTON,
        'B+shift': LABEL_BACK_BUTTON,
        'D+shift': LABEL_DISCARD_BUTTON,
        'S+shift': LABEL_SKIP_BUTTON,
        'S+shift+alt': SKIP_BATCH_BUTTON,
    }
}

# Option 1 have a Hotkey component in each page layout
# Avoid having to swap map keyEvent -> key id
# Options 2 Have one global Hotkey component
def setup_hotkeys():
    # Load configured
    return [
        Keyboard(
            id=KEYBOARD_ID
        ),

        dcc.Store(id=HOTKEY_CFG_ID, storage_type='local', data=dict()),
        dcc.Store(id=CAPTURE_KEY_TRIGGER),
        dcc.Store(id='blur-input')
    ]


@callback(
    Input('url', 'pathname'),
    output=dict(
        capture_keys=Output(KEYBOARD_ID, 'captureKeys'),
        active_key_map=Output(HOTKEY_CFG_ID, 'data'),
    ),
    prevent_initial_call=False
)
def on_page_changed(
    path
):
    print('on page changed')

    path = path.split("/")[1]
    key_map = hotkeys.get(path)

    captureKeys = _extract_capture_keys(key_map)
    print(f'captureKeys: {captureKeys}')
    print(f'keymap: {key_map}')
    return dict(
        capture_keys=captureKeys,
        active_key_map=key_map,
    )

@callback(
    Input(KEYBOARD_ID, 'n_keydowns'),
    State(KEYBOARD_ID, 'keydown'),
    State(HOTKEY_CFG_ID, 'data'),
    # output=dict(
    #     btn_id=Output(CAPTURE_KEY_TRIGGER, 'data')
    # )
)
def on_capture_key_pressed(
    trigger,
    key_event,
    hotkey_map,
):
    if trigger is None or key_event['repeat']:
        raise PreventUpdate

    print(json.dumps(key_event))

    cord_id = key_cort_dict_to_str(key_event)
    print(cord_id)
    button_id = hotkey_map.get(cord_id, None)

    if button_id is None:
        # not a mapped binding
        raise PreventUpdate

    # simulate button click
    set_props(
        button_id,
        dict(n_clicks=0)
    )

    set_props(
        'blur-input',
        dict(data=True)
    )

def _extract_capture_keys(hotkeys: dict) -> list:
    capture_keys = []
    for combo in hotkeys:
        key = combo.split('+')[0]
        capture_keys.append(key)
    return capture_keys

def key_cort_dict_to_str(key_event: dict) -> str:
    key, alt, ctrl, shift, meta, _ = key_event.values()
    print(key_event.values())
    return (
        f'{key}'
        f'{"+shift" if shift else ""}'
        f'{"+alt" if alt else ""}'
    )


# TODO RENAME function
clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='blurSearchInput'),
    Input('blur-input', 'data'),
    prevent_initial_call=True,
)


