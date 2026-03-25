from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
import logging

from dash import (
    ClientsideFunction,
    Dash,
    Input,
    Output,
    State,
    clientside_callback,
    set_props,
    ALL,
)
import dash
from dash.exceptions import PreventUpdate

from skactiveml_annotation.core import api
from skactiveml_annotation.shared_ids import (
    FOCUS_ELEMENT_TRIGGER,
    SELECTION,
    BATCH_STATE,
    BROWSER_DATA,
)
from skactiveml_annotation.ui import common
from skactiveml_annotation.ui.pages.annotation import components
from skactiveml_annotation.ui.pages.annotation.label_setting_modal import SortBySetting
from skactiveml_annotation.ui.pages.annotation.modality import (
    DataDisplaySetting,
    create_data_display,
    create_data_presentation_settings,
)

from skactiveml_annotation.core.schema import (
    Batch,
    Annotation,
    AnnotationMetaData,
    AnnotationProgress,
    SessionConfig,
    DISCARD_MARKER,
    MISSING_LABEL_MARKER,
)

from skactiveml_annotation.ui.pages.home.selection import Selection

from . import (
    ids,
    actions,
)


ALL_ANNOTATION_BTNS = {'type': ids.ACTION_BTN, 'index': ALL}


def register(app: Dash):
    @app.callback(
        Input(ids.ANNOTATION_INIT, 'pathname'),
        State(SELECTION, 'data'),
        State(BATCH_STATE, 'data'),
        State(ids.BATCH_SIZE_INPUT, 'value'),
        output=dict(
            ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
            query_trigger=Output(ids.QUERY_TRIGGER, 'data', allow_duplicate=True),
            annot_progress=Output(ids.ANNOT_PROGRESS, 'data'),
            data_presentation_setting_children=Output(ids.DATA_PRESENTATION_SETTINGS_CONTAINER, "children"),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def init(
        _,
        selection: Selection,
        batch: Batch | None,
        batch_size: int,
    ):
        annot_progress = _init_annot_progress(selection)
        activeml_cfg = common.compose_from_state(selection)


        if annot_progress.is_all_annotated():
            if batch is None:
                # Restore batch
                history_idx = api.get_global_history_idx(activeml_cfg.dataset.id)
                batch = api.restore_batch(activeml_cfg, history_idx, False, batch_size)

            must_query = False
        else:
            must_query = (
                batch is None
                or batch.is_completed()
            )

        if must_query:
            ui_trigger = dash.no_update
            query_trigger = True
        else:
            ui_trigger = True
            query_trigger = dash.no_update

        modality = activeml_cfg.dataset.modality
        api.ensure_global_history_idx_init(activeml_cfg.dataset.id)

        return dict(
            ui_trigger=ui_trigger,
            query_trigger=query_trigger,
            annot_progress=annot_progress,
            data_presentation_setting_children=create_data_presentation_settings(modality),
            batch=batch,
        )
    _ = init


    @app.callback(
        Input(actions.CONFIRM.btn_id, 'n_clicks'),
        Input(actions.DISCARD.btn_id, 'n_clicks'),
        Input(actions.SKIP.btn_id, 'n_clicks'),
        State(SELECTION, 'data'),
        State(BATCH_STATE, 'data'),
        State(ids.TIME_STAMP, 'data'),
        State(ids.LABEL_CHIPS_INPUT, 'value'),
        State(ids.ANNOT_PROGRESS, 'data'),
        output=dict(
            annot_data=Output(ids.ANNOT_PROGRESS, 'data', allow_duplicate=True),
            search_text=Output(ids.LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
            focus_trigger=Output(FOCUS_ELEMENT_TRIGGER, "data", allow_duplicate=True),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def on_confirm(
        confirm_click: int | None,
        discard_click: int | None,
        skip_click: int | None,
        selection: Selection,
        batch: Batch,
        time_stamp_str: str,
        value: str,
        annot_progress: AnnotationProgress,
    ):
        end_time = datetime.now(timezone.utc)

        if all(x is None for x in (confirm_click, discard_click, skip_click)):
            raise PreventUpdate

        try:
            trigger_id = common.get_trigger_id()
        except RuntimeError as e:
            logging.error(e)
            raise PreventUpdate

        label = (
            MISSING_LABEL_MARKER if trigger_id == 'skip' else
            value if trigger_id == 'confirm' else
            DISCARD_MARKER
        )

        start_time = datetime.fromisoformat(time_stamp_str)
        annotation = batch.get_annotation()
        annot_metadata = _init_or_update_annot_metadata(annotation, start_time, end_time)

        if trigger_id == "skip":
            annot_metadata.skip_intended_cnt += 1

        annotation = Annotation(
            embedding_idx=batch.get_emb_index(),
            label=label,
            meta_data=annot_metadata,
        )
        batch.add_annotation(annotation)

        if not batch.is_advanceable(step=1):
            logging.info("Batch is completed")

            dataset_id = selection.dataset_id
            embedding_id = selection.embedding_id

            api.update_json_annotations(dataset_id, embedding_id, batch)
            annot_progress.num_annotated = api.get_num_annotated(dataset_id, exclude_missing=True)

            if annot_progress.is_all_annotated():
                logging.info("All Samples Annotated!")
                annot_progress.num_annotated
                set_props(ids.UI_TRIGGER, dict(data=True))
            else:
                set_props(ids.QUERY_TRIGGER, dict(data=True))
        else:
            batch.advance(step=1)
            set_props(ids.UI_TRIGGER, dict(data=True))

        return dict(
            annot_data=annot_progress,
            search_text='',
            focus_trigger=ids.LABEL_SEARCH_INPUT,
            batch=batch,
        )
    _ = on_confirm


    @app.callback(
        Input(ids.UI_TRIGGER, 'data'),
        State(SELECTION, 'data'),
        State(BATCH_STATE, 'data'),
        State(ids.DATA_DISPLAY_CFG_DATA, 'data'),
        State(BROWSER_DATA, 'data'),
        State(ids.ADDED_CLASS_NAME, 'data'),
        State(ids.LABEL_SETTING_SHOW_PROBAS, 'checked'),
        State(ids.LABEL_SETTING_SORTBY, 'value'),
        State(ALL_ANNOTATION_BTNS, 'id'),
        output=dict(
            label_container=Output(ids.LABELS_CONTAINER, 'children'),
            show_container=Output(ids.DATA_DISPLAY_CONTAINER, 'children'),
            batch_progress=Output('batch-progress-bar', 'value'),
            data_display_data=Output(ids.DATA_DISPLAY_CFG_DATA, 'data'),
            is_computing_overlay=Output(ids.COMPUTING_OVERLAY, 'visible', allow_duplicate=True),
            data_width=Output(ids.DATA_DISPLAY_CONTAINER, 'w'),
            data_height=Output(ids.DATA_DISPLAY_CONTAINER, 'h'),
            annot_start_time_trigger=Output(ids.START_TIME_TRIGGER, 'data'),
            disable_all_action_buttons=Output(ALL_ANNOTATION_BTNS, 'loading', allow_duplicate=True),
            focus_trigger=Output(FOCUS_ELEMENT_TRIGGER, "data"),
            added_class_name=Output(ids.ADDED_CLASS_NAME, 'data'),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True,
    )
    def on_ui_update(
        # Triggers
        ui_trigger: bool | None,
        # Data
        selection: Selection,
        batch: Batch,
        data_display_setting: DataDisplaySetting | None,
        browser_dpr,
        # Adding classes
        added_class_name: str | None,
        # Label settings
        show_probas: bool,
        sort_by: str,
        annot_button_ids: list,
    ):
        if ui_trigger is None and browser_dpr is None:
            raise PreventUpdate

        activeml_cfg = common.compose_from_state(selection)
        modality = activeml_cfg.dataset.modality

        annotation = batch.get_annotation()

        human_data_path = Path(
            api.get_one_file_path(
                activeml_cfg.dataset.id,
                activeml_cfg.embedding.id,
                batch.get_emb_index(),
            )
        )

        if data_display_setting is None:
            data_display_setting = DataDisplaySetting()

        rendered_data, w, h = create_data_display(
            data_display_setting,
            modality,
            human_data_path,
            browser_dpr,
        )

        sort_by = SortBySetting[sort_by]

        return dict(
            label_container=components.create_label_chips(
                activeml_cfg.dataset.classes, annotation, batch, show_probas, sort_by, added_class_name
            ),
            show_container=rendered_data,
            batch_progress=batch.get_progress_percent(),
            data_display_data=data_display_setting,
            is_computing_overlay=False,
            data_width=w,
            data_height=h,
            annot_start_time_trigger=True,
            disable_all_action_buttons=[False] * len(annot_button_ids),
            focus_trigger=ids.LABEL_SEARCH_INPUT,
            added_class_name=None,
            batch=batch,
        )
    _ = on_ui_update


    @app.callback(
        Input(ids.QUERY_TRIGGER, 'data'),
        State(SELECTION, 'data'),
        State(ids.BATCH_SIZE_INPUT, 'value'),
        State(ids.SUBSAMPLING_INPUT, 'value'),
        output=dict(
            ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True,
        # background=True, # LRU Cache won't work with this
    )
    def on_next_batch(
        trigger: bool | None,
        selection: Selection,
        batch_size: int,
        subsampling: int | float,
    ):
        if trigger is None:
            raise PreventUpdate

        # Assumes global idx is on the last of the completed batch
        # to determine correct number of restorable samples
        session_cfg = SessionConfig(batch_size, subsampling)
        activeml_cfg = common.compose_from_state(selection)
        dataset_id = activeml_cfg.dataset.id

        global_history_idx = api.get_global_history_idx(dataset_id)
        history_size = api.get_num_annotated(dataset_id)
        num_restorable = max(0, history_size - (global_history_idx + 1))

        if num_restorable >= batch_size:
            # No Active ML needed just restore Batch size many samples
            batch = api.restore_batch(activeml_cfg, global_history_idx, True, batch_size)

            # This assumes the idx is on the last of the previous batch
            api.increment_global_history_idx(dataset_id, 1)

        else:
            # Active learning needed. But first restore what is left to restore
            if num_restorable > 0:
                batch_one = api.restore_batch(activeml_cfg, global_history_idx, True, num_restorable)
                emb_indices_one = batch_one.emb_indices
                api.increment_global_history_idx(dataset_id, 1)

                # Only the difference has to be queried
                session_cfg.batch_size = batch_size - num_restorable

                X = api.load_embeddings(activeml_cfg.dataset.id, activeml_cfg.embedding.id)

                # Remove samples from pool that have been restored. To avoid possible duplication
                batch_two = api.request_query(activeml_cfg, session_cfg, X, emb_indices_one)

                batch = batch_one.concat(batch_two)

            else:
                new_history_idx = api.get_num_annotated(dataset_id)
                api.set_global_history_idx(dataset_id, new_history_idx)

                session_cfg.batch_size = batch_size - num_restorable

                X = api.load_embeddings(activeml_cfg.dataset.id, activeml_cfg.embedding.id)
                batch = api.request_query(activeml_cfg, session_cfg, X)

        return dict(
            ui_trigger=True,
            batch=batch,
        )
    _ = on_next_batch


    @app.callback(
        Input(ids.SKIP_BATCH_BTN, 'n_clicks'),
        State(SELECTION, 'data'),
        State(BATCH_STATE, 'data'),
        State(ids.ANNOT_PROGRESS, 'data'),
        output=dict(
            query_trigger=Output(ids.QUERY_TRIGGER, 'data'),
            annot_progress=Output(ids.ANNOT_PROGRESS, 'data', allow_duplicate=True),
            search_text=Output(ids.LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
            focus_trigger=Output(FOCUS_ELEMENT_TRIGGER, "data", allow_duplicate=True),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def on_skip_batch(
        n_clicks: int,
        selection: Selection,
        batch: Batch,
        annot_progress: AnnotationProgress,
    ):
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate

        dataset_id = selection.dataset_id
        embedding_id = selection.embedding_id

        api.update_json_annotations(dataset_id, embedding_id, batch)
        annot_progress.num_annotated = api.get_num_annotated(dataset_id, exclude_missing=True)

        return dict(
            query_trigger=True,
            annot_progress=annot_progress,
            search_text='',
            focus_trigger=ids.LABEL_SEARCH_INPUT,
            batch=None, # Reset batch so a new one is computed
        )
    _ = on_skip_batch


    @app.callback(
        Input(actions.BACK.btn_id, 'n_clicks'),
        State(BATCH_STATE, 'data'),
        State(SELECTION, 'data'),
        State(ids.TIME_STAMP, 'data'),
        State(ids.BATCH_SIZE_INPUT, 'value'),
        State(ids.ANNOT_PROGRESS, 'data'),
        output=dict(
            ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
            annot_progress=Output(ids.ANNOT_PROGRESS, 'data', allow_duplicate=True),
            search_text=Output(ids.LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
            focus_trigger=Output(FOCUS_ELEMENT_TRIGGER, "data", allow_duplicate=True),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def on_back(
        clicks: None | int,
        batch: Batch,
        selection: Selection,
        time_stamp_str: str,
        batch_size: int,
        annot_progress: AnnotationProgress,
    ):
        end_time = datetime.now(timezone.utc)

        if clicks is None:
            raise PreventUpdate

        start_time = datetime.fromisoformat(time_stamp_str)
        annotation = batch.get_annotation()
        annot_metadata = _init_or_update_annot_metadata(annotation, start_time, end_time)

        old_label = (
            MISSING_LABEL_MARKER if annotation is None else
            annotation.label
        )

        annotation = Annotation(
            embedding_idx=batch.get_emb_index(),
            label=old_label,
            meta_data=annot_metadata,
        )
        batch.add_annotation(annotation)

        if batch.progress == 0:
            dataset_id = selection.dataset_id
            embedding_id = selection.embedding_id
            file_paths = api.get_file_paths(dataset_id, embedding_id, batch.emb_indices)
            api.update_annotations(dataset_id, file_paths, batch.annotations)

            activeml_cfg = common.compose_from_state(selection)
            history_idx = api.get_global_history_idx(activeml_cfg.dataset.id)

            try:
                batch = api.restore_batch(activeml_cfg, history_idx, False, batch_size)
            except RuntimeError:
                logging.warning("Raise PreventUpdate. Cannot go back further. No Annotations left")
                # Have to do ui_trigger so the buttons are enabled
                return dict(
                    ui_trigger=True,
                    store_data=dash.no_update,
                    annot_progress=dash.no_update,
                    search_text=dash.no_update,
                    focus_trigger=dash.no_update,
                    batch=dash.no_update,
                )

            api.increment_global_history_idx(dataset_id, -len(batch))

            # Annotations can decrease if a previous annotation of was changed to SKIP
            annot_progress.num_annotated = api.get_num_annotated(dataset_id, exclude_missing=True)

        else:
            batch.advance(step= -1)

        return dict(
            ui_trigger=True,
            annot_progress=annot_progress.model_dump(),
            search_text='',
            focus_trigger=ids.LABEL_SEARCH_INPUT,
            batch=batch,
        )
    _ = on_back


    @app.callback(
        Input(ids.UI_TRIGGER, 'data'),
        State(ids.ANNOT_PROGRESS, 'data'),
        output=dict(
            num_annotated=Output(ids.ANNOT_PROGRESS_TEXT, 'value'),
            num_samples=Output(ids.NUM_SAMPLES_TEXT, 'value'),
        ),
        prevent_initial_call=True
    )
    def on_annot_progress_change(
        trigger: bool | None,
        annot_progress: AnnotationProgress,
    ):
        if trigger is None:
            raise PreventUpdate

        return dict(
            num_annotated=annot_progress.num_annotated,
            num_samples=annot_progress.num_samples,
        )
    _ = on_annot_progress_change


    @app.callback(
        Input(ids.START_TIME_TRIGGER, 'data'),
        output=dict(
            now_str=Output(ids.TIME_STAMP, 'data', allow_duplicate=True)
        ),
        prevent_initial_call=True
    )
    def on_annot_start_timestamp(
        trigger: bool | None,
    ):
        if trigger is None:
            raise PreventUpdate

        # TODO: this will be utc aware time of the server and not user
        return dict(
            now_str=datetime.now(timezone.utc).isoformat()
        )
    _ = on_annot_start_timestamp


    @app.callback(
        Input(ids.ADD_CLASS_BTN, 'n_clicks'),
        State(BATCH_STATE, 'data'),
        State(SELECTION, 'data'),
        State(ids.LABEL_SEARCH_INPUT, 'value'),
        output=dict(
            ui_trigger=Output(ids.UI_TRIGGER, 'data', allow_duplicate=True),
            search_value=Output(ids.LABEL_SEARCH_INPUT, 'value', allow_duplicate=True),
            added_class_name=Output(ids.ADDED_CLASS_NAME, 'data', allow_duplicate=True),
            label_value=Output(ids.LABEL_CHIPS_INPUT, 'value', allow_duplicate=True),
            batch=Output(BATCH_STATE, 'data', allow_duplicate=True),
        ),
        prevent_initial_call=True
    )
    def on_add_new_class(
        click: int | None,
        batch: Batch,
        selection: Selection,
        new_class_name: str | None,
    ):
        if click is None:
            raise PreventUpdate

        if new_class_name is None:
            logging.warning("Failed to add class because no class name is provided")
            raise PreventUpdate

        activeml_cfg = common.compose_from_state(selection)

        try:
            api.add_class(
                dataset_cfg=activeml_cfg.dataset,
                new_class_name=new_class_name,
                batch=batch,
            )
        except ValueError as e:
            logging.warning(f"Failed to add class: {e}")
            raise PreventUpdate

        return dict(
            ui_trigger=True,
            search_value='',
            added_class_name=new_class_name,
            label_value=new_class_name,
            batch=batch,
        )
    _ = on_add_new_class


    # Get initial browser config like dpr.
    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='getDpr'),
        Output(BROWSER_DATA, 'data'),
        Input(ids.ANNOTATION_INIT, 'pathname')
    )

    # Disable buttons to prevent spamming before processing is done.
    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='disableAllButtons'),
        Output(ALL_ANNOTATION_BTNS, 'loading'),
        Input(ALL_ANNOTATION_BTNS, 'n_clicks'),
        prevent_initial_call=True
    )

    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='scrollToChip'),
        Output(ids.LABEL_CHIPS_INPUT, 'value'),
        Input(ids.LABEL_SEARCH_INPUT, "value"),
        prevent_initial_call=True,
    )

    # On Query start. Show loading overlay.
    clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='triggerTrue'),
        Output(ids.COMPUTING_OVERLAY, 'visible'),
        Input(ids.QUERY_TRIGGER, 'data')
    )


def _init_annot_progress(selection: Selection) -> AnnotationProgress:
    return AnnotationProgress(
        num_annotated=api.get_num_annotated(selection.dataset_id, exclude_missing=True),
        num_samples=api.get_total_num_samples(selection.dataset_id, selection.embedding_id)
    )


def _init_or_update_annot_metadata(
    old_annotation: Annotation | None,
    start_time: datetime,
    end_time: datetime,
    updated_label: str | None = None,
) -> AnnotationMetaData:
    delta_time = end_time - start_time

    if old_annotation is None:
        # New Annotation
        return AnnotationMetaData(
            first_view_time=start_time,
            total_view_duration=delta_time,
            last_edit_time=end_time,
        )

    old_meta = old_annotation.meta_data
    label_changed = old_annotation.label != updated_label
    return AnnotationMetaData(
        first_view_time=old_meta.first_view_time,
        total_view_duration=old_meta.total_view_duration + delta_time,
        last_edit_time=end_time if label_changed else old_meta.last_edit_time,
        skip_intended_cnt=old_meta.skip_intended_cnt,
    )
