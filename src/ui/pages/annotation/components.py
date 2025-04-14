import numpy as np

import dash_mantine_components as dmc

from core.schema import DataType
from ui.pages.annotation.data_display import *


def create_sidebar():
    return (
        dmc.Stack(
            [
                dmc.Center(
                    dmc.Text("Settings"),
                ),

                # Batch Size selection
                dmc.NumberInput(
                    label="Batch Size",
                    id='batch-size-input',
                    allowNegative=False,
                    debounce=True,
                    value=5,
                    required=True,
                    persistence='batch-size-persistence',
                    persistence_type='local',
                ),

                # Subsampling selection
                dmc.NumberInput(
                    label="Subsampling",
                    id='subsampling-input',
                    allowNegative=False,
                    debounce=True,
                    hideControls=True
                ),

                dmc.Text(
                    'Query Strategy'
                ),

                # Skip Button
                dmc.Center(
                    dmc.Button(
                        "Skip Batch",
                        id="skip-batch-button",
                    ),
                ),
            ],
            style={'border': '2px solid red'},
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
            style={'border': 'red dashed 2px'},
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
                    radius=25,
                    size="lg",
                    style={"height": "40px"},
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
                        "color": "white",
                        "pointerEvents": "none",
                    },
                ),
            ],
            style={"position": "relative", "width": "100%"},
        )
    )


def create_data_display(data_type, human_data_path):
    if data_type.value == DataType.IMAGE.value:
        rendered_data = create_image_display(human_data_path)
    elif data_type.value == DataType.TEXT.value:
        rendered_data = create_text_display(human_data_path)
    else:
        rendered_data = create_audio_display(human_data_path)

    return (
        dmc.Stack(
            rendered_data,
            style={'border': '4px dotted pink'},
            align="center",
        )
    )


# TODO bad name
def create_chip_group(classes, batch):
    # Check if there is some annotation already for that sample in case the user used back btn.
    annotation = batch.annotations[batch.progress]
    was_annotated = annotation is not None
    was_labaled = was_annotated and isinstance(annotation, int)

    class_prob = None
    if batch.class_probas:
        class_prob = batch.class_probas[batch.progress]

    if class_prob is None:
        chips = [_create_chip(idx, label) for idx, label in enumerate(classes)]
        preselect = str(annotation) if was_labaled else None
    else:
        if was_labaled:
            preselect = str(annotation)
        elif was_annotated:
            preselect = None
        else:
            highest_prob_idx = np.argmax(class_prob)
            preselect = str(highest_prob_idx)

        chips = [_create_chip(idx, label, probability) for idx, (label, probability) in
                 enumerate(zip(classes, class_prob))]

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
                    'flex-direction': 'row',
                    'flex-wrap': 'wrap',
                    'gap': '10px'
                }
            ),
        ),
        id='my-scroll-area',
        type='auto',
        offsetScrollbars=True,
        styles=dict(
            viewport={
                'max-height': '35vh'
            },
            border='green dashed 3px',
        ),
        style={
            'border': 'green dashed 3px'
        },
        w='50vw'
    )


def _create_chip(idx, label, probability=None):
    chip = dmc.Chip(
        label,
        id=f'chip-{idx}',
        value=str(idx),
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


