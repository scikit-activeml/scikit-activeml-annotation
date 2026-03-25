from pathlib import Path

from dash import dcc
import dash_mantine_components as dmc

from ._model import TextDataDisplaySetting

from .. import ids

TEXT_FONT_SIZE_INPUT = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'text', 'index': 'font_size' }
TEXT_LINE_HEIGHT_INPUT = { 'type': ids.DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'text', 'index': 'line_height' }


def display(path: Path, text_display_setting: TextDataDisplaySetting):
    if not path.exists():
        raise ValueError(f"Cannot load text data from path: {path}")

    text_data = path.read_text(encoding="utf-8").strip()

    return (
        dmc.ScrollArea(
            dmc.Box(
                dcc.Markdown(
                    text_data,
                    style={
                        "margin": "0",
                        "padding": "0",
                        "fontSize": f"{text_display_setting.font_size}px",
                        "fontFamily": "'Consolas', 'Courier New', 'Lucida Console', monospace",
                        "fontVariantLigatures": "none",  # disable ligatures

                        "lineHeight": text_display_setting.line_height,
                        # Preserve line breaks but wrap long lines
                        "whiteSpace": "pre-line",  # allow wrapping (default is normal)
                        "wordBreak": "normal",  # break long words if needed
                        "overflowWrap": "normal", # ensures text doesn't overflow
                        # Width of the Text Container
                        "overflowX": "hidden", # no horizontal scrollbar
                        # 'border':'blue dotted 2px',
                    }
                ),
                py=1,
                style={
                    "width": "46vw",        # restrict component width
                    "overflowX": "hidden",      # no horizontal scrollbar
                    # "border": "2px solid green",  # Debug border
                },
            ),
            type='auto',
            offsetScrollbars='y',
            styles=dict(
                viewport={
                    'maxHeight': '60vh',
                    # 'border':'brown dashed 3px',
                },
            )
        )
    )


def presentation_settings():
    default_text_setting = TextDataDisplaySetting()

    return (
        dmc.Stack(
            [
                dmc.NumberInput(
                    id=TEXT_FONT_SIZE_INPUT,
                    min=1,
                    max=35,
                    step=1,
                    clampBehavior='strict',
                    hideControls=False,
                    decimalScale=2,
                    label="Font size",
                    placeholder=str(default_text_setting.font_size),
                    value=default_text_setting.font_size,
                    allowNegative=False,
                    # w='35%',
                    persistence='font-size-persistence',
                    persistence_type='session'
                ),
            
                dmc.NumberInput(
                    id=TEXT_LINE_HEIGHT_INPUT,
                    min=0.2,
                    max=35,
                    step=0.1,
                    clampBehavior='strict',
                    hideControls=False,
                    decimalScale=2,
                    label="Line height",
                    placeholder=str(default_text_setting.line_height),
                    value=default_text_setting.line_height,
                    allowNegative=False,
                    # w='35%',
                    persistence='line-height-persistence',
                    persistence_type='session'
                ),
            ],
            align='start'
        )
    )
