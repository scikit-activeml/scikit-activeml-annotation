
from collections import defaultdict
from typing import Any, Final, Literal
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
)

import dash
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc

from skactiveml_annotation.core.api import camel_case_to_title
from skactiveml_annotation.shared_ids import GO_LAST_PAGE_TRIGGER, KEYMAPPING_CFG
from skactiveml_annotation.ui.hotkeys import (
    HotkeyConfig,
    button_actions,
    normalize_hotkey_str,
)

from . import (
    ids,
)


def register(app: Dash):
    @app.callback(
        Input(ids.UI_UPDATE_TRIGGER, "data"),
        State(KEYMAPPING_CFG, "data"),
        output=dict(
            hotkey_cfg_container=Output(ids.CONFIGURATION_CONTAINER, "children")
        ),
        prevent_initial_call=True
    )
    def update_hotkey_page(_, hotkey_cfg: HotkeyConfig):
        mapping = hotkey_cfg.mapping

        content = dmc.Stack(
            [_build_page_ui(page, modal_mapping)
            for page, modal_mapping in mapping.items()]
        )

        return dict(hotkey_cfg_container=content)
    _ = update_hotkey_page


    @app.callback(
        Input(ids.URL_INIT, "pathname"),
        output=dict(
            ui_trigger=Output(ids.UI_UPDATE_TRIGGER, "data")
        )
    )
    def init_hotkey_page(
        _,
    ):
        return dict(
            ui_trigger=dict(data=True)
        )
    _ = init_hotkey_page


    @app.callback(
        Input(ids.CONFIRM_HOTKEYS_BTN, "n_clicks"),
        State({"type": ids.HOTKEY_INPUT, "action": ALL}, "id"),
        State({"type": ids.HOTKEY_INPUT, "action": ALL}, "value"),
        output=dict(
            hotkey_cfg=Output(KEYMAPPING_CFG, "data", allow_duplicate=True),
            ui_trigger=Output(ids.UI_UPDATE_TRIGGER, "data", allow_duplicate=True),
            errors=Output({"type": ids.HOTKEY_INPUT, "action": ALL}, "error"),
        ),
        prevent_initial_call=True
    )
    def on_hotkey_cfg_change_confirmed(
        n_clicks: int | None,
        ids: list[dict[str, Any]],
        updated_hotkeys: list[Any],
    ):
        if n_clicks is None:
            raise PreventUpdate

        num_inputs: Final = len(ids)

        errors: list[str | Literal[False]] = [False] * num_inputs

        # Create a 2-level nested defaultdict:
        # updated_cfg[key1][key2] is always initialized to an empty dict.
        # See: https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict
        updated_cfg = defaultdict(lambda: defaultdict(dict))

        for idx, (entry_id, new_hotkey) in enumerate(zip(ids, updated_hotkeys)):
            action_id = entry_id['action']

            # i.e. Page.Modal.Confirm
            page, modal, _ = action_id.split(".")

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

        return dict(
            hotkey_cfg=HotkeyConfig(
                mapping=updated_cfg,
                is_user_defined=True,
            ).model_dump(),
            ui_trigger=dict(data=True),
            errors=[False] * num_inputs,
        )
    _  = on_hotkey_cfg_change_confirmed


    @app.callback(
        Input(ids.RESET_HOTKEYS_BTN, "n_clicks"),
        output=dict(
            hotkey_cfg=Output(KEYMAPPING_CFG, "data", allow_duplicate=True),
            ui_trigger=Output(ids.UI_UPDATE_TRIGGER, "data", allow_duplicate=True)
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
    _ = reset_hotkeys_to_default


    @app.callback(
        Input(ids.BACK_BTN, 'n_clicks'),
        output=dict(
            go_last_page_trigger=Output(GO_LAST_PAGE_TRIGGER, "data"),
        ),
        prevent_initial_call=True
    )
    def on_back(
        clicks: int | None
    ):
        if clicks is None:
            raise PreventUpdate

        return dict(
            go_last_page_trigger=True,
        )
    _ = on_back



def _build_page_ui(page_name: str, modal_mapping: dict):
    """Return a Stack for a full page with modals."""
    return dmc.Stack(
        [
            dmc.Text(f"{page_name} Page", size="lg"),
            dmc.Stack(
                [
                    _build_modal_ui(modal, key_mapping) 
                    for modal, key_mapping in modal_mapping.items()
                ]
            ),
        ]
    )


def _build_modal_ui(modal_name: str, key_mapping: dict):
    """Return a Stack for a modal inside a page."""
    return dmc.Stack(
        [
            dmc.Text(
                camel_case_to_title(modal_name),
                size="md",
            ) if modal_name != "Main" else None,
            _build_key_mapping_ui(key_mapping),
        ]
    )


def _build_key_mapping_ui(key_mapping: dict):
    """Return a Stack of key inputs for a modal."""
    return dmc.Stack(
        [
            dmc.Flex(
                [
                    dmc.TextInput(
                        value=key_combo,
                        id={"type": ids.HOTKEY_INPUT, "action": action_id},
                    ),
                    dmc.Box(button_actions()[action_id].btn_text),
                ],
                gap="md",
                align="center",
            )
            for key_combo, action_id in key_mapping.items()
        ]
    )
