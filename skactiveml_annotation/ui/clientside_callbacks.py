from dash import (
    ClientsideFunction,
    Input,
    clientside_callback,
)

from skactiveml_annotation.shared_ids import (
    CLICK_BTN_TRIGGER,
    FOCUS_ELEMENT_TRIGGER,
    GO_LAST_PAGE_TRIGGER,
)


def register():
    # Trigger to simulate a button click
    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='clickButtonWithId'),
        Input(CLICK_BTN_TRIGGER, "data"),
    )

    # Trigger to focus an element with an id
    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='focusElementWithId'),
        Input(FOCUS_ELEMENT_TRIGGER, "data"),
    )

    # Trigger to go back to the last page
    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='goToLastPage'),
        Input(GO_LAST_PAGE_TRIGGER, "data"),
    )
