from typing import Protocol, Sequence
import logging

from dash import (
    ClientsideFunction,
    Dash,
    Input,
    Output,
    State,
    ctx,
    set_props,
    clientside_callback,
)
import dash
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from dash_iconify import DashIconify

from skactiveml_annotation.core import api
from skactiveml_annotation.hydra_schema import DatasetConfig
from skactiveml_annotation.shared_ids import (
    SELECTION,
    BATCH_STATE,
    FOCUS_ELEMENT_TRIGGER,
)
from skactiveml_annotation.ui.components import sampling_input
from skactiveml_annotation.ui.pages.home.selection import (
    Selection,
    SelectionProgress,
    SelectionStep,
)

from . import (
    ids,
)


class HasIdAndDisplayName(Protocol):
    id: str
    display_name: str


def register(app: Dash):
    @app.callback(
        Input(ids.CONFIRM_BUTTON, 'n_clicks'),
        Input(ids.BACK_BUTTON, 'n_clicks'),
        Input(ids.STEPPER, 'active'),
        State(ids.RADIO_SELECTION, 'value'),
        State(ids.STEPPER, 'active'),
        State(ids.SELECTION_PROGRESS, 'data'),
        output=dict(
            selection_content=Output(ids.UI_CONTAINER, 'children', allow_duplicate=True),
            selection=Output(ids.SELECTION_PROGRESS, 'data', allow_duplicate=True),
            step=Output(ids.STEPPER, 'active', allow_duplicate=True),
            focus=Output(FOCUS_ELEMENT_TRIGGER, 'data', allow_duplicate=True),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def update(
        confirm_clicks: int | None,
        back_clicks: int | None,
        new_active: int | None,
        radio_value: str,
        current_step: int,
        selection: SelectionProgress | None,
    ):
        if (
            selection is None
            or (confirm_clicks is None and back_clicks is None and new_active is None)
        ):
            raise PreventUpdate

        trigger_id = ctx.triggered_id

        if trigger_id == ids.CONFIRM_BUTTON:
            return _handle_confirm(radio_value, current_step, selection)

        elif trigger_id == ids.BACK_BUTTON:
            return _handle_back(current_step, selection)

        elif trigger_id == ids.STEPPER:
            if new_active is None:
                logging.error("new_active is None when the trigger was the STEPPER. Should not occure")
                raise PreventUpdate
            return _handle_ui_stepper_clicked(new_active, selection)
    _ = update


    @app.callback(
        Input(ids.URL_INIT, 'pathname'),
        State(ids.SELECTION_PROGRESS, 'data'),
        output=dict(
            selection_content=Output(ids.UI_CONTAINER, 'children', allow_duplicate=True),
            selection=Output(ids.SELECTION_PROGRESS, 'data', allow_duplicate=True),
        ),
        prevent_initial_call='initial_duplicate',
    )
    def setup_page(
        _,
        selection_data: SelectionProgress | None,
    ):
        if selection_data is None:
            selection = SelectionProgress()
        else:
            selection = selection_data

        return dict(
            selection_content=_create_step_ui(SelectionStep(0), selection),
            selection=selection,
        )
    _ = setup_page


    @app.callback(
        Input(ids.NEXT_PAGE_TRIGGER, 'data'),
        State(ids.SELECTION_PROGRESS, 'data'),
        output=dict(
            pathname=Output(ids.URL, 'pathname'),
            selection=Output(SELECTION, 'data', allow_duplicate=True),
        ),
        initial_duplicate=True,
    )
    def go_to_next_page(
        trigger: bool | None,
        selection_progress: SelectionProgress,
    ):
        if trigger is None:
            raise PreventUpdate

        selection = selection_progress.convert()

        if api.is_dataset_embedded(selection.dataset_id, selection.embedding_id):
            pathname = f'/annotation/{selection.dataset_id}'
        else:
            pathname = f'/embedding'

        return dict(
            pathname=pathname,
            selection=selection,
        )
    _ = go_to_next_page


    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='validateConfirmButton'),
        Output(ids.CONFIRM_BUTTON, "disabled"),
        Input(ids.RADIO_SELECTION, "value"),
    )


def _handle_confirm(
    radio_value: str,
    current_step: int,
    selection: SelectionProgress,
):
    batch = dash.no_update

    if current_step >= Selection.size():
        set_props(ids.NEXT_PAGE_TRIGGER, dict(data=True))
        return dict(
            selection_content=dash.no_update,
            selection=dash.no_update,
            step=dash.no_update,
            focus=dash.no_update,
            batch=batch,
        )

    elif current_step == 0:
        prev_dataset_id = selection.get(SelectionStep.DATASET)
        was_dataset_changed = prev_dataset_id is not None and radio_value != prev_dataset_id
        if was_dataset_changed:
            batch = None

    selection.add(SelectionStep(current_step), radio_value)
    new_step = SelectionStep(current_step + 1)

    return dict(
        selection_content=_create_step_ui(new_step, selection),
        selection=selection,
        step=new_step,
        focus=ids.UI_CONTAINER,
        batch=batch,
    )


def _handle_back(
    current_step: int,
    selection: SelectionProgress,
):
    if current_step == 0:
        raise PreventUpdate

    next_step = SelectionStep(current_step - 1)

    return dict(
        selection_content=_create_step_ui(next_step, selection),
        selection=dash.no_update,
        step=next_step,
        focus=ids.UI_CONTAINER,
        batch=dash.no_update,
    )


def _handle_ui_stepper_clicked(
    new_active: int,
    selection: SelectionProgress,
):
    new_active = SelectionStep(new_active)
    return dict(
        selection_content=_create_step_ui(new_active, selection),
        selection=dash.no_update,
        step=dash.no_update,
        focus=ids.UI_CONTAINER,
        batch=dash.no_update,
    )


# Helper function to build UI for different steps
def _create_step_ui(step: SelectionStep, selection: SelectionProgress):
    if step == SelectionStep.DATASET:
        preselect = selection.get(SelectionStep.DATASET)
        content = _create_dataset_selection(preselect)
    elif step == SelectionStep.EMBEDDING:
        content = _create_embedding_radio_group(selection)
    elif step == SelectionStep.QUERY:
        content = _create_radio_group(api.get_qs_config_options(), selection.get(SelectionStep.QUERY))
    elif step == SelectionStep.MODEL:
        content = _create_radio_group(api.get_model_config_options(), selection.get(SelectionStep.MODEL))
    elif step == SelectionStep.SAMPLING_PARAM:
        content = dmc.Stack(
            [
                *sampling_input.create_sampling_inputs(),
                # Dummy element to ensure this id exists in the layout at the last step
                dmc.RadioGroup([], id=ids.RADIO_SELECTION, display='none', readOnly=True)
            ],
            align="flex-start"
        )

    return dmc.ScrollArea(
        content,
        offsetScrollbars='y',
        type='auto',
        scrollbars='y',
        styles=dict(
            viewport={
                'maxHeight': '90%'
            },
        )
    )


def _create_dataset_radio_item(cfg: DatasetConfig, cfg_display: str):
    dataset_exists = api.dataset_path_exits(cfg.data_path)

    radio_item = (
        dmc.Radio(
            label=cfg_display,
            value=cfg.id,
            disabled=not dataset_exists,
            size='md',
        )
    )

    if dataset_exists:
        return radio_item

    return dmc.Tooltip(
        radio_item,
        label=f'Dataset does not exist at path: {cfg.data_path}',
        openDelay=250,
    )


def _create_dataset_selection(preselect: str | None):
    dataset_options = api.get_dataset_config_options()
    data = [(cfg, f'{cfg.display_name} - ({cfg.modality.value})')
            for cfg in dataset_options]

    return (
        dmc.RadioGroup(
            id=ids.RADIO_SELECTION,
            children=dmc.Stack(
                [
                    _create_dataset_radio_item(cfg, cfg_display)
                    for cfg, cfg_display in data
                ]
            ),
            value=preselect,
            size="md",
            wrapperProps={"autoFocus": True},
            p="xs",
            # style={'border': '2px solid red'}
        )
    )


def _create_embedding_radio_group(selection: SelectionProgress):
    dataset_cfg_id = selection.get_not_none(SelectionStep.DATASET)

    modality = api.get_dataset_cfg_from_id(dataset_cfg_id).modality

    # Only show embedding methods applicable to the modality of the dataset
    options = [
        emb for emb in api.get_embedding_config_options()
        if modality in emb.modalities
    ]

    formatted_options = [(cfg.id, cfg.display_name) for cfg in options]

    preselect = selection.get(SelectionStep.EMBEDDING)

    return dmc.RadioGroup(
        id=ids.RADIO_SELECTION,
        children=dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.Radio(label=cfg_name, value=cfg_id, size='md'),
                        _create_bool_icon(api.is_dataset_embedded(
                            selection.get_not_none(SelectionStep.DATASET),
                            cfg_id
                        ))
                    ]
                )
                for cfg_id, cfg_name in formatted_options
            ]
        ),
        value=preselect,
        size="md",
        p="xs",
        # style={'border': '2px solid red'},
    )


def _create_bool_icon(val: bool):
    if val:
        icon = 'tabler:check'
        color = 'green'
        label = 'embedding is cached'
    else:
        icon = 'tabler:x'
        color = 'red'
        label = 'embedding has to be computed'

    return (
        dmc.Tooltip(
            dmc.ThemeIcon(
                DashIconify(icon=icon),
                variant='light',
                radius=20,
                color=color,
                size=25,
            ),
            label=label
        )
    )


def _create_radio_group(
    options: Sequence[HasIdAndDisplayName],
    preselect: str | None,
):
    formatted_options = [(cfg.id, cfg.display_name) for cfg in options]
    return dmc.RadioGroup(
        id=ids.RADIO_SELECTION,
        children=dmc.Stack([dmc.Radio(label=l, value=k, size='md') for k, l in formatted_options]),
        value=preselect,
        size="md",
        p="xs",
        # style={'border': '2px solid red'},
    )
