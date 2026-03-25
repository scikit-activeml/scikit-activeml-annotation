import logging

import numpy as np

import dash_mantine_components as dmc

from skactiveml_annotation.core.schema import (
    Annotation,
    Batch,
    MISSING_LABEL_MARKER,
)
from skactiveml_annotation.ui.components import sampling_input

from . import (
    ids,
    actions,
)
from .label_setting_modal import SortBySetting


def create_sidebar():
    return (
        dmc.Stack(
            [
                dmc.Stack(
                    [
                        dmc.Center(
                            dmc.Title("Settings", order=3),
                        ),

                        *sampling_input.create_sampling_inputs(),
                    ],
                    # style={'border': 'red 3px dotted'},
                    gap=10,
                    mb=10,
                ),

                dmc.Divider(variant="solid"),

                dmc.Stack(
                    [
                        dmc.Center(
                            dmc.Title("Presentation", order=3)
                        ),
                        dmc.Center(
                            dmc.ScrollArea(
                                dmc.Center(
                                    dmc.Box(
                                        id=ids.DATA_PRESENTATION_SETTINGS_CONTAINER,
                                        mih=15,
                                        my=10,
                                        # style=dict(border='4px dotted red')
                                    ),
                                ),
                                type='auto',
                                offsetScrollbars='y',
                                styles=dict(
                                    viewport={
                                        'maxHeight': '20vh'
                                    },
                                    border='green dashed 3px',
                                ),
                                style={
                                    # 'border': 'green dashed 3px'
                                },
                                w='50vw'
                            )
                        ),


                        dmc.Center(
                            dmc.Tooltip(
                                dmc.Button(
                                    "Apply",
                                    id=actions.APPLY.btn_id,
                                    color='dark'
                                ),
                                label="Apply Presentation Settings"
                            )
                        )
                    ],
                    gap=10,
                    my=10,
                ),

                dmc.Divider(variant="solid"),

                dmc.Stack(
                    [
                        dmc.Center(
                            dmc.Title("Actions", order=3)
                        ),

                        # Skip Button
                        dmc.Center(
                            dmc.HoverCard(
                                [
                                    dmc.HoverCardTarget(
                                        dmc.Button(
                                            "Skip Batch",
                                            id=ids.SKIP_BATCH_BTN,
                                            color='dark'
                                        ),
                                    ),

                                    dmc.HoverCardDropdown(
                                        dmc.Center(
                                            dmc.Text(
                                                'Write back all annotated samples in this batch. Skip the rest. '
                                                'Then recompute next batch with current configuration. ',
                                                maw='10vw'
                                            )
                                        )
                                    )
                                ],
                                openDelay=500,
                                shadow='lg',
                            )
                        ),

                        dmc.Center(
                            dmc.Button(
                                'Auto Annotate',
                                id=ids.AUTO_ANNOTATE_BTN,
                                color='dark'
                            )
                        )

                    ],
                    # style={'border': 'red 3px dotted'},
                    gap=15,
                    mb=10
                ),
            ],
            p='xs',
            # mt=15,
            gap=10,
            # style={'border': '2px solid blue'},
        )
    )


def create_confirm_buttons():
    return (
        dmc.Group(
            [
                dmc.Button(
                    actions.BACK.btn_text,
                    id=actions.BACK.btn_id,
                    color='dark'
                ),

                dmc.Button(
                    actions.DISCARD.btn_text,
                    id=actions.DISCARD.btn_id,
                    color='dark'
                ),

                dmc.Button(
                    actions.SKIP.btn_text,
                    id=actions.SKIP.btn_id,
                    color='dark'
                ),

                dmc.Button(
                    actions.CONFIRM.btn_text,
                    id=actions.CONFIRM.btn_id,
                    color='dark',
                ),
            ],
            justify='center',
            # style={'border': 'red dashed 2px'},
            # gap=20
        )
    )


def create_progress_bar(progress=0):
    return (
        dmc.Box(
            [
                # The Mantine Progress bar with dynamic section
                dmc.ProgressRoot(
                    dmc.ProgressSection(
                        id='batch-progress-bar',
                        value=progress * 100,
                        color="blue",
                        # animated=True,
                        # striped=True
                    ),
                    transitionDuration=500,
                    radius=8,
                    size="lg",
                    style={"height": "25px"},
                ),
                # The overlay text: always centered
                dmc.Box(
                    "Batch Progress",
                    style={
                        "position": "absolute",
                        "width": "100%",
                        "top": "50%",
                        "left": "50%",
                        "transform": "translate(-50%, -50%)",
                        "textAlign": "center",
                        "color": "black",
                        "pointerEvents": "none",
                    },
                ),
            ],
            style={
                "position": "relative",
                "width": "50vw",
                # 'border': 'gold dotted 3px'
            },
        )
    )


def create_label_chips(
    classes_yaml: list[str],
    annotation: Annotation | None,
    batch: Batch,
    show_probas: bool,
    sort_by: SortBySetting,
    preselect: str | None,
):
    # Check if there is some annotation already for that sample in case the user used back btn.
    was_annotated = annotation is not None

    class_probas = None
    if batch.class_probas is not None:
        class_probas = batch.class_probas[batch.progress]

    if class_probas is not None and show_probas:
        # Sorted classes and class_probas
        classes_yaml, class_probas = _sort(classes_yaml, batch.classes_sklearn, class_probas, sort_by)
        chips = [_create_chip(label, probability) for label, probability in
                 zip(classes_yaml, class_probas)]
    else:
        chips = [_create_chip(label) for label in classes_yaml]

    # Determine which label to preselect
    if preselect is not None:
        logging.info(f"preselect after adding label: {preselect}")
    elif was_annotated:
        # Was allready previously annoated. For intance when going back
        preselect = annotation.label
    elif class_probas is not None:
        highest_prob_idx = np.argmax(class_probas)
        preselect = classes_yaml[int(highest_prob_idx)]
    else:
        preselect = MISSING_LABEL_MARKER

    chip_group = dmc.ChipGroup(
        children=chips,
        multiple=False,
        value=preselect,
        id=ids.LABEL_CHIPS_INPUT,
    )

    return dmc.ScrollArea(
        dmc.Center(
            dmc.Box(
                chip_group,
                style={
                    'display': 'inline-flex',
                    'flexDirection': 'row',
                    'flexWrap': 'wrap',
                    'gap': '10px',
                },
                py=5,
            ),
        ),
        id='my-scroll-area',
        type='auto',
        offsetScrollbars='y',
        styles=dict(
            viewport={
                'maxHeight': '35vh'
            },
            # border='green dashed 3px',
        ),
        style={
            # 'border': 'green dashed 3px'
        },
        w='50vw',
    )


def _sort(
    classes_yaml: list[str],
    classes_sklearn: list[str],
    class_probas: list[float],
    sort_by: SortBySetting
) -> tuple[list[str], list[float]]:
    """
    Sorts classes and probabilities according to the user's preference.

    Parameters
    ----------
    classes_yaml : list[str]
        Original class order defined by the user (YAML).
    classes_sklearn : list[str]
        scikit-learn internal class order (clf.classes_).
    class_probas : list[float] | None
        Probabilities from clf.predict_proba (aligned with classes_sklearn).
    sort_by : SortBySetting

    Returns
    -------
    tuple[list[str], list[float]]
        The sorted class names and corresponding probabilities.
    """
    match sort_by:
        case SortBySetting.yaml_order:
            # Return in YAML-defined order
            # Need to remap from sklearn's order -> YAML order
            mapping = {cls: i for i, cls in enumerate(classes_sklearn)}
            sorted_indices = [mapping[cls] for cls in classes_yaml if cls in mapping]

        case SortBySetting.proba:
            sorted_indices = sorted(range(len(class_probas)), key=lambda i: class_probas[i], reverse=True)

        case SortBySetting.alphabet:
            # sklearn already ensures alphabetical order so just return as is
            return classes_sklearn, class_probas
        
    return (
        [classes_sklearn[i] for i in sorted_indices],
        [class_probas[i] for i in sorted_indices]
    )


def _create_chip(label: str, probability: float | None =None):
    label = label.strip()

    # the javascript clientside callback 'scrollToChip' requires that
    # the id is of format 'chip-{label}' and value=label to function correctly
    chip = dmc.Chip(
        label,
        id=f'chip-{label}',
        value=label,
        # Ensures text inside label is centered
        styles={"label": {"textAlign": "center"}},
    )
    if probability is None:
        return chip

    return dmc.InputWrapper(
        chip,
        inputWrapperOrder=['input', 'label', 'description'],
        description=f"{probability:.2f}",
        style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
    )


