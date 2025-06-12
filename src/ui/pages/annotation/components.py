import logging
import numpy as np

import dash_mantine_components as dmc
from omegaconf import OmegaConf

from core.schema import DataType, MISSING_LABEL_MARKER
from ui.components.sampling_input import create_sampling_inputs
from ui.pages.annotation.data_display import *

from ui.pages.annotation.ids import *
from ui.pages.annotation.label_setting_modal import SORT_BY_PROBA, SORT_BY_ALPHABET


def create_sidebar():
    return (
        dmc.Stack(
            [
                dmc.Stack(
                    [
                        dmc.Center(
                            dmc.Title("Settings", order=3),
                        ),

                        *create_sampling_inputs(),

                        # Data display settings Button
                        dmc.Center(
                            dmc.Tooltip(
                                dmc.Button(
                                    'Display Settings',
                                    id=DATA_DISPLAY_BTN,
                                    color='dark',
                                    mt=15
                                ),
                                label="Change how data is displayed",
                                openDelay=500
                            ),
                        ),
                    ],
                    # style={'border': 'red 3px dotted'},
                    gap=10,
                    mb=10,
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
                                            id=SKIP_BATCH_BUTTON,
                                            color='dark'
                                        ),
                                    ),

                                    dmc.HoverCardDropdown(
                                        dmc.Center(
                                            dmc.Text(
                                                'Write back all annotated samples in this batch. Skip the rest. '
                                                'Then recompute next batch with current configuration. ',
                                                maw='10vw'  # TODO hardcoded
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
                                id=AUTO_ANNOTATE_BTN,
                                color='dark'
                            )
                        )

                    ],
                    # style={'border': 'red 3px dotted'},
                    gap=10
                ),

                # TODO allow to switch Query Strategy during annotation.
                # dmc.Text(
                #     'Query Strategy'
                # ),
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
                    'Back',
                    id=LABEL_BACK_BUTTON,
                    color='dark'
                ),

                dmc.Button(
                    'Discard',
                    id=LABEL_DISCARD_BUTTON,
                    color='dark'
                ),

                dmc.Button(
                    'Skip',
                    id=LABEL_SKIP_BUTTON,
                    color='dark'
                ),

                dmc.Button(
                    'Confirm',
                    id=LABEL_CONFIRM_BUTTON,
                    color='dark'
                ),

            ],
            # style={'border': 'red dashed 2px'},
            justify='center',
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


def create_data_display(data_type, human_data_path, dpr):
    if data_type.value == DataType.IMAGE.value:
        rendered_data, w, h = create_image_display(human_data_path, dpr)
    elif data_type.value == DataType.TEXT.value:
        rendered_data = create_text_display(human_data_path)
    else:
        rendered_data = create_audio_display(human_data_path)

    return (
        rendered_data,
        w,
        h
    )


def create_label_chips(classes, batch, show_probas, sort_by, was_class_added, insertion_idxes):
    # Check if there is some annotation already for that sample in case the user used back btn.
    annotation = batch.annotations[batch.progress]
    was_annotated = annotation is not None

    class_prob = None
    if batch.class_probas:
        class_prob = batch.class_probas[batch.progress]

        if insertion_idxes is not None:
            # Some classes have been added for which the classifier has not yet been fitted for.
            # Set the class_prob to 0.0 in this case for these classes
            class_prob = _pad_with_zeros(class_prob, insertion_idxes)

    elif show_probas:
        logging.info("Cannot show probas")

    sorted_classes, sorted_probas = _sort(classes, class_prob, sort_by)

    has_probas = class_prob is not None
    show_probas = has_probas and show_probas

    # Determine preselect
    if was_class_added:
        # preselect last added class
        preselect = classes[insertion_idxes[-1]]
        logging.info(f"preselect after adding label: {preselect}")
    elif was_annotated:
        preselect = annotation
    elif has_probas:
        highest_prob_idx = np.argmax(sorted_probas)
        preselect = sorted_classes[int(highest_prob_idx)]
    else:
        preselect = MISSING_LABEL_MARKER

    if show_probas:
        chips = [_create_chip(label, probability) for label, probability in
                 zip(sorted_classes, sorted_probas)]
    else:
        chips = [_create_chip(label) for label in sorted_classes]

    chip_group = dmc.ChipGroup(
        children=chips,
        multiple=False,
        value=preselect,
        id="label-radio",
    )

    return dmc.ScrollArea(
        dmc.Center(
            dmc.Box(
                chip_group,
                style={
                    'display': 'inline-flex',
                    'flexDirection': 'row',
                    'flexWrap': 'wrap',
                    'gap': '10px'
                }
            ),
        ),
        id='my-scroll-area',
        type='auto',
        offsetScrollbars=True,
        styles=dict(
            viewport={
                'maxHeight': '35vh'
            },
            # border='green dashed 3px',
        ),
        style={
            # 'border': 'green dashed 3px'
        },
        w='50vw'
    )


def _pad_with_zeros(class_probas, insertion_idxes):
    """
    Return a new list that has the same entries as class_probas,
    but with a 0 inserted at each position in insertion_idxs.
    """
    new_length = len(class_probas) + len(insertion_idxes)

    result = [0.0] * new_length

    orig_i = 0
    for i in range(new_length):
        if i not in insertion_idxes:
            result[i] = class_probas[orig_i]
            orig_i += 1

    return result


def _sort(classes, class_probas, sort_by):
    if sort_by == SORT_BY_PROBA:
        if class_probas is None:
            logging.warning("Cannot sort by predicted class probabilities as this info is not available.")
            return classes, class_probas

        sorted_indices = sorted(range(len(class_probas)), key=lambda i: class_probas[i], reverse=True)
    elif sort_by == SORT_BY_ALPHABET:
        sorted_indices = sorted(range(len(classes)), key=lambda i: str.lower(classes[i]))

    else:
        return classes, class_probas

    return (
        [classes[i] for i in sorted_indices],
        [class_probas[i] for i in sorted_indices]
    )


def _create_chip(label, probability=None):
    chip = dmc.Chip(
        label,
        id=f'chip-{label}',
        value=label,
        styles={"label": {"textAlign": "center"}},  # Ensures label is centered
    )
    if probability is None:
        return chip

    return dmc.InputWrapper(
        chip,
        inputWrapperOrder=['input', 'label', 'description'],
        # description=dmc.Text(f"{probability:.2f}", size="xs"),
        description=f"{probability:.2f}",
        style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
    )


