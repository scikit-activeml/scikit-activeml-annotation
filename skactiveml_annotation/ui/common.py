import logging

import pydantic

from dash.exceptions import PreventUpdate

from skactiveml_annotation.core.api import compose_config
from skactiveml_annotation.core.schema import ActiveMlConfig
from skactiveml_annotation.ui.hotkeys import HotkeyConfig
from skactiveml_annotation.ui.storekey import StoreKey

def compose_from_state(store_data) -> ActiveMlConfig:
    overrides = (
        ('dataset', store_data[StoreKey.DATASET_SELECTION.value]),
        ('query_strategy', store_data[StoreKey.QUERY_SELECTION.value]),
        ('embedding', store_data[StoreKey.EMBEDDING_SELECTION.value]),
        ('+model', store_data[StoreKey.MODEL_SELECTION.value])  # add model to default list
    )

    return compose_config(overrides)


def try_deserialize_hotkey_cfg(hotkey_cfg_json) -> HotkeyConfig:
    if hotkey_cfg_json is None:
        logging.error(
            "Hotkey Config should be initialized allready but it is None"
        )
        raise PreventUpdate

    try:
        hotkey_cfg = HotkeyConfig.model_validate(hotkey_cfg_json)
    except pydantic.ValidationError as e:
        logging.error("Unexpected deserialization error occured \n%s", e)
        raise PreventUpdate

    return hotkey_cfg
