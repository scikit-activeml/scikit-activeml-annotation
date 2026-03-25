from dash_extensions import Keyboard

from dash import (
    Dash,
    dcc,
    register_page,
)

import dash_mantine_components as dmc

import dash_loading_spinners as dls
from dash_iconify import DashIconify

from skactiveml_annotation.shared_ids import BATCH_STATE

from . import (
    ids,
    actions,
    callbacks,
    components,
    auto_annotate_modal,
    label_setting_modal,
    modality,
)


def register(app: Dash):
    register_page(
        __name__,
        path_template='/annotation/<dataset_name>',
        layout=_layout,
        description='The main annotation page',
    )

    actions.register(app)
    callbacks.register(app)
    label_setting_modal.register_callbacks(app)
    auto_annotate_modal.register_callbacks(app)
    modality.register_callbacks(app)


def _layout(**kwargs):
    _ = kwargs
    return(
        dmc.Box(
            [
                dcc.Location(id=ids.ANNOTATION_INIT, refresh=False),
                # Triggers
                dcc.Store(id=ids.UI_TRIGGER),
                dcc.Store(id=ids.QUERY_TRIGGER),
                dcc.Store(id=ids.START_TIME_TRIGGER),
                # Data
                dcc.Store(id=ids.TIME_STAMP, storage_type="session"),
                dcc.Store(id=ids.DATA_DISPLAY_CFG_DATA, storage_type='session'),
                dcc.Store(id=ids.ANNOT_PROGRESS, storage_type='session'),
                dcc.Store(id=ids.ADDED_CLASS_NAME, storage_type='session'),

                Keyboard(id=ids.KEYBOARD),

                label_setting_modal.create_label_settings_modal(),
                auto_annotate_modal.create_auto_annotate_modal(),

                dmc.Box(id=ids.LABEL_CHIPS_INPUT),  # avoid id error

                dmc.AppShell(
                    [
                        dmc.AppShellNavbar(
                            children=components.create_sidebar(),
                            p="md",
                            # style={'border': '4px solid red'}
                        ),
                        dmc.Flex(
                            [
                                dmc.Box(
                                    [
                                        dmc.LoadingOverlay(
                                            id=ids.COMPUTING_OVERLAY,
                                            zIndex=10,
                                            loaderProps={
                                                'children': dmc.Stack(
                                                    [
                                                        dmc.Group(
                                                            [
                                                                dmc.Title("Computing next batch", order=2),
                                                                # Show 3 dots duruing query
                                                                dmc.Loader(
                                                                    size='xl',
                                                                    type='dots',
                                                                    # Type annotation incorrect valid css is supported
                                                                    color='var(--mantine-color-dark-7)',  # pyright: ignore[reportArgumentType]
                                                                ),
                                                            ],
                                                            justify='center',
                                                            wrap='wrap',
                                                            mb='5vh'  # pyright: ignore[reportArgumentType]
                                                        ),
                                                    ],
                                                    align='center',
                                                    # style=dict(border='red dashed 3px')
                                                )
                                            },
                                            overlayProps={
                                                'radius':'lg',
                                                'center': True,
                                                'blur': 7,
                                            },
                                            transitionProps={
                                                'transition':'fade',
                                                'duration': 150,
                                                'mounted': True,
                                                # 'exitDuration': 500,
                                            },
                                        ),

                                        dmc.Center(
                                            dcc.Loading(
                                                dmc.Box(
                                                    id=ids.DATA_DISPLAY_CONTAINER,
                                                    mih=15,
                                                    my=10,
                                                    # style=dict(border='4px dotted red')
                                                ),
                                                delay_hide=150,
                                                delay_show=150,
                                                custom_spinner=dls.ThreeDots(radius=7)
                                            ),
                                        ),

                                        dmc.Group(
                                            [
                                                dmc.Tooltip(
                                                    dmc.ActionIcon(
                                                        DashIconify(icon='tabler:plus',width=20),
                                                        variant='filled',
                                                        id=ids.ADD_CLASS_BTN,
                                                        color="dark"
                                                    ),
                                                    label="Add a new class by using current Search Input."
                                                ),

                                                dmc.Tooltip(
                                                    dmc.ActionIcon(
                                                        DashIconify(icon="clarity:settings-line", width=20),
                                                        variant="filled",
                                                        id=ids.LABEL_SETTING_BTN,
                                                        color='dark',
                                                    ),
                                                    label='Label settings',
                                                ),

                                                components.create_confirm_buttons(),

                                                dmc.TextInput(
                                                    placeholder='Select Label',
                                                    id=ids.LABEL_SEARCH_INPUT,
                                                    radius='sm',
                                                    w='150px',
                                                    # inputProps={
                                                    #     "autoFocus": True
                                                    # }
                                                ),
                                            ],
                                            mt=15,
                                            justify='center'
                                        ),
                                    ],
                                    p=10,
                                    pos="relative",
                                ),

                                dmc.Stack(
                                    id=ids.LABELS_CONTAINER,
                                    # h='400px'
                                    align='center'
                                ),
                                # create_confirm_buttons(),
                                components.create_progress_bar()
                            ],


                            style={
                                # 'border': '5px dotted blue',
                                'height': '100%',
                                'widht': '100%',
                            },
                            justify='center',
                            align='center',
                            direction='column',
                            wrap='nowrap',
                            gap=10,
                            py=0,
                            px=150,
                        ),

                        dmc.AppShellAside(
                            children=[
                                dmc.Stack(
                                    [
                                        dmc.Card(
                                            dmc.Stack(
                                                [
                                                    dmc.Center(
                                                        dmc.Title("Stats", order=3)
                                                    ),

                                                    dmc.Tooltip(
                                                        dmc.Group(
                                                            [
                                                                dmc.Text("Annotated:", style={"fontSize": "1vw"}),
                                                                dmc.Text(
                                                                    dmc.NumberFormatter(
                                                                        id=ids.ANNOT_PROGRESS_TEXT,
                                                                        thousandSeparator=' ',
                                                                    ),
                                                                    style={"fontSize": "1vw"}
                                                                ),
                                                            ],
                                                            gap=4
                                                        ),
                                                        label="Number of samples annotated."
                                                    ),

                                                    dmc.Tooltip(
                                                        dmc.Group(
                                                            [
                                                                dmc.Text("Total:", style={"fontSize": "1vw"}),
                                                                dmc.Text(
                                                                    dmc.NumberFormatter(
                                                                        id=ids.NUM_SAMPLES_TEXT,
                                                                        thousandSeparator=' '
                                                                    ),
                                                                    style={"fontSize": "1vw"}
                                                                )
                                                            ],
                                                            gap=4
                                                        ),
                                                        label='Total number of samples in dataset'
                                                    )

                                                ],
                                                gap=5
                                            )
                                        )
                                    ],
                                    # p='xs',
                                    # style={'border': '3px dashed green'},
                                    align='center'
                                )
                            ],
                            p="xs",
                            # style={'border': '4px solid red'}
                        ),
                    ],
                    navbar={  # pyright: ignore[reportArgumentType]
                        "width": '13vw',
                        "breakpoint": "sm",
                        "collapsed": {"mobile": True},
                    },
                    aside={  # pyright: ignore[reportArgumentType]
                        "width": '13vw',
                        "breakpoint": "sm",
                        "collapsed": {"mobile": True},
                    },
                    padding=0,
                ),
            ],
            style={
                # 'height': '100%',
                # 'border': 'green dotted 5px'
            }
        )
    )
