from dash import (
    Input,
    Output,
    State,
    callback, ClientsideFunction, clientside_callback,
)

from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc

from core.api import auto_annotate, load_embeddings, save_partial_annotations
from core.schema import Batch
from ui.common import compose_from_state
from ui.pages.annotation.ids import *
from ui.storekey import StoreKey, AnnotProgress


def create_auto_annotate_modal():
    return dmc.Modal(
        dmc.Stack(
            [

                dmc.NumberInput(
                    id=AUTO_ANNOTATE_THRESHOLD,
                    min=0,
                    max=1,
                    hideControls=True,
                    label="Threshold",
                    placeholder="Enter a threshold",
                    value=0.99,
                    allowNegative=False,
                    w='35%',
                    required=True,
                    persistence='auto-annotate-threshold-persistence',
                    persistence_type='local',
                ),

                dmc.Center(
                    dmc.Button(
                        'Confirm',
                        id=AUTO_ANNOTATE_CONFIRM_BTN,
                        color='dark',
                    )
                )
            ],
        ),
        id=AUTO_ANNOTATE_MODAL,
        title='Auto Annotate with Threshold',
        centered=True,
        shadow='xl',
    )


@callback(
    Input(AUTO_ANNOTATE_BTN, 'n_clicks'),
    output=dict(
        modal_open=Output(AUTO_ANNOTATE_MODAL, 'opened'),
    ),
    prevent_initial_call=True
)
def open_modal(
    clicks
):
    if clicks is None:
        raise PreventUpdate

    return dict(
        modal_open=True
    )


# TODO this should be a background callback
@callback(
    Input(AUTO_ANNOTATE_CONFIRM_BTN, 'n_clicks'),
    State('session-store', 'data'),
    State(ANNOT_PROGRESS, 'data'),
    State(AUTO_ANNOTATE_THRESHOLD, 'value'),
    # output=dict(
    #     # annot_progress=Output(ANNOT_PROGRESS, 'data', allow_duplicate=True)
    #     # query_trigger=Output(QUERY_TRIGGER, 'data', allow_duplicate=True)
    # ),
    prevent_initial_call=True,
)
def on_auto_annotate(
    click,
    session_data,
    annot_progress,
    threshold
):
    if click is None:
        raise PreventUpdate

    # TODO what happens with the current batch Write back all annoted before doing it?

    activeml_cfg = compose_from_state(session_data)
    X = load_embeddings(
        activeml_cfg.dataset.id,
        activeml_cfg.embedding.id
    )

    # TODO some duplicate code
    batch_json = session_data.pop(StoreKey.BATCH_STATE.value, None)
    dataset_id = session_data[StoreKey.DATASET_SELECTION.value]
    embedding_id = session_data[StoreKey.EMBEDDING_SELECTION.value]
    batch = Batch.from_json(batch_json)

    # TODO updating annot_progress does not trigger ui update!
    # num_annotated = save_partial_annotations(batch, dataset_id, embedding_id)
    # annot_progress[AnnotProgress.PROGRESS.value] = num_annotated

    auto_annotate(X, activeml_cfg, threshold)

    # return dict(
    #     # query_trigger=True
    #     # annot_progress=annot_progress
    # )


# Close Modal on confirm
# TODO should the modal be closed during computation or after?
# clientside_callback(
#     ClientsideFunction(namespace='clientside', function_name='false'),
#     Output(AUTO_ANNOTATE_MODAL, 'opened'),
#     Input(AUTO_ANNOTATE_CONFIRM_BTN, 'n_clicks')
# )



