# TODO: Rename these are not real components
# It should be clear from the name if defines callbacks or no
import logging
from pathlib import Path

import numpy as np

import dash_mantine_components as dmc

from skactiveml_annotation.core.schema import (
    Batch, 
    DataType, 
    MISSING_LABEL_MARKER
)
from skactiveml_annotation.ui.components import sampling_input

from . import ids
from . import data_display
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

                        # Data display settings Button
                        dmc.Center(
                            dmc.Tooltip(
                                dmc.Button(
                                    'Display Settings',
                                    id=ids.DATA_DISPLAY_BTN,
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
                                            id="skip-batch-button",
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
                                id=ids.AUTO_ANNOTATE_BTN,
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
                    id="back-button",
                    color='dark'
                ),

                dmc.Button(
                    'Discard',
                    id='discard-button',
                    color='dark'
                ),

                dmc.Button(
                    'Skip',
                    id="skip-button",
                    color='dark'
                ),

                dmc.Button(
                    'Confirm',
                    id='confirm-button',
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


def create_data_display(data_type, human_data_path: Path, dpr):
    if data_type == DataType.IMAGE:
        rendered_data, w, h = data_display.create_image_display(human_data_path, dpr)
    elif data_type == DataType.TEXT:
        rendered_data, w, h = data_display.create_text_display(human_data_path)
    else:
        rendered_data, w, h = data_display.create_audio_display(human_data_path)

    return (
        rendered_data,
        w,
        h
    )


# TODO: missing type annotation
def create_label_chips(
    classes: list[str], 
    # TODO: The batch does not always contain probas as some classifers dont have this method
    batch: Batch,  
    show_probas: bool, 
    sort_by: SortBySetting,
    was_class_added: bool, 
    insertion_idxes: list[int]
):
    # Check if there is some annotation already for that sample in case the user used back btn.
    annotation = batch.annotations[batch.progress]
    was_annotated = annotation is not None # TODO: Using None if its not annotated acctually made sense

    class_probas = None
    has_probas = False
    if batch.class_probas is not None:
        class_probas = batch.class_probas[batch.progress]
        has_probas = True

    # Check if probabilities have to be sorted
    must_sort = (
        class_probas is not None
        and (show_probas or (not was_class_added and not was_annotated))
    )

    if must_sort:
        if insertion_idxes is not None:
            # Some classes have been added for which the classifier has not yet been fitted for.
            # Set the class_prob to 0.0 in this case for these classes
            class_probas = _pad_with_zeros(class_probas, insertion_idxes)

        # Sorted classes and class_probas
        classes, class_probas = _sort(classes, class_probas, sort_by)

    if was_class_added:
        # preselect last added class
        preselect = classes[insertion_idxes[-1]]
        logging.info(f"preselect after adding label: {preselect}")
    elif was_annotated:
        # Was allready previously annoated. For intance when going back
        preselect = annotation
    elif class_probas is not None:
        highest_prob_idx = np.argmax(class_probas)
        preselect = classes[int(highest_prob_idx)]
    else:
        preselect = MISSING_LABEL_MARKER


    if show_probas and class_probas is not None:
        chips = [_create_chip(label, probability) for label, probability in
                 zip(classes, class_probas)]
    else:
        chips = [_create_chip(label) for label in classes]


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


# TODO: It might be worth to convert to numpy array honestly
def _sort(classes: list[str], class_probas: list[float], sort_by: SortBySetting):
    if sort_by == SortBySetting.proba:
        if class_probas is None:
            logging.warning("Cannot sort by predicted class probabilities as this info is not available.")
            return classes, class_probas

        sorted_indices = sorted(range(len(class_probas)), key=lambda i: class_probas[i], reverse=True)
    elif sort_by == SortBySetting.alphabet:
        sorted_indices = sorted(range(len(classes)), key=lambda i: str.lower(classes[i]))

    elif sort_by == SortBySetting.no_sort:
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


