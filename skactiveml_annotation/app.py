import dash
from dash import (
    Dash,
    Input,
    Output,
    State,
    html,
    dcc,
    callback,
    DiskcacheManager,
)
import diskcache
from dash.exceptions import PreventUpdate

from dash_extensions.enrich import (
    DashProxy,
    BaseModelTransform,
)

import dash_mantine_components as dmc
import dash_loading_spinners

from skactiveml_annotation.shared_ids import (
    BROWSER_DATA,
    CLICK_BTN_TRIGGER,
    FOCUS_ELEMENT_TRIGGER,
    GO_LAST_PAGE_TRIGGER,
    KEYMAPPING_CFG,
    SELECTION,
    BATCH_STATE,
)
from skactiveml_annotation.ui import clientside_callbacks
from skactiveml_annotation.ui.components import navbar
import skactiveml_annotation.paths as sap
from skactiveml_annotation.ui import hotkeys
from skactiveml_annotation.ui.pages import (
    annotation,
    home,
    embedding,
    hotkeys_cfg,
)

cache = diskcache.Cache(sap.BACKGROUND_CALLBACK_CACHE_PATH)
background_callback_manager = DiskcacheManager(cache)

PAGE_CONTENT_CONTAINER = "page_content_container"
APP_SPINNER_CONTAINER = "app_spinner_container"

def create_app() -> Dash:
    # dash_extensions.enrich.DashProxy is used because it allows the use of
    # pydantic models as argument and return types of dash callbacks
    # and therefore reduces manual serde inside callbacks
    app = DashProxy(
        __package__,
        use_pages=True,  # Use dash page feature
        pages_folder="", # Manually register pages instead of dash doing it.
        external_stylesheets=[dmc.theme.DEFAULT_THEME] + dmc.styles.ALL,
        # Allows to register callbacks on components that will be created by other callbacks,
        # and are therefore not in the initial layout.
        suppress_callback_exceptions=True,
        # prevent_intial_callback is not correctly types in dash_extensions
        prevent_initial_callbacks=True,  # pyright: ignore[reportArgumentType]
        assets_folder=str(sap.ASSETS_PATH),
        title="scikit-activeml-annotation",
        background_callback_manager=background_callback_manager,
        update_title='',
        transforms=[BaseModelTransform()]
    )

    app.layout = layout
    
    clientside_callbacks.register()
    register_pages(app)

    return app


def register_pages(app):
    home.layout.register(app)
    annotation.layout.register(app)
    embedding.layout.register(app)
    hotkeys_cfg.layout.register(app)

    hotkeys.register_callbacks(app)


def layout(**kwargs):
    _ = kwargs
    return (
        dmc.MantineProvider(
            dmc.AppShell(
                [
                    # Data stored across all pages
                    dcc.Store(BROWSER_DATA),
                    dcc.Store(SELECTION, storage_type='session'),
                    dcc.Store(id=BATCH_STATE, storage_type="session"),

                    # Triggers
                    dcc.Store(CLICK_BTN_TRIGGER),
                    dcc.Store(FOCUS_ELEMENT_TRIGGER),
                    dcc.Store(GO_LAST_PAGE_TRIGGER),

                    # Hotkeys per Page
                    dcc.Store(KEYMAPPING_CFG, storage_type="local"),

                    navbar.create_navbar(),
                    dmc.AppShellMain(
                        [
                            html.Div(
                                dash_loading_spinners.Pacman(
                                    fullscreen=True,
                                ),
                                id=APP_SPINNER_CONTAINER
                            ),

                            dmc.Container(
                                dash.page_container,
                                id=PAGE_CONTENT_CONTAINER,
                                fluid=True,
                                style={'padding': 0}
                            )
                        ],
                        style={
                            # 'border': '5px dashed red',
                            # 'height': '80%'
                        },
                    )
                ],
                header={'height': 50},
            )
        )
    )


@callback(
    Output(APP_SPINNER_CONTAINER, 'children'),
    Input(PAGE_CONTENT_CONTAINER, 'loading_state'),
    State(APP_SPINNER_CONTAINER, 'children'),
)
def hide_page_loading_spinner(
    _,
    children,
):
    if children:
        return None
    raise PreventUpdate
