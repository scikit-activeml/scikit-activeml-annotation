import dash
from dash import (
    Dash,
    Input,
    Output,
    State,
    html,
    dcc,
    callback
)
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc
import dash_loading_spinners

from werkzeug.middleware.profiler import ProfilerMiddleware

from ui.components.navbar import create_navbar
from paths import (
    PAGES_PATH,
    PROFILER_PATH,
    ASSETS_PATH
)

app = Dash(
    __name__, 
    use_pages=True,  # Use dash page feature
    pages_folder=str(PAGES_PATH),
    external_stylesheets=[dmc.theme.DEFAULT_THEME, dmc.styles.ALL],
    # Allows to register callbacks on components that will be created by other callbacks,
    # and are therefore not in the initial layout.
    suppress_callback_exceptions=True, 
    prevent_initial_callbacks=True,
    assets_folder=str(ASSETS_PATH),
    title="scikit-activeml-annotation",
)


app.layout = (
    dmc.MantineProvider(
        dmc.AppShell(
            [
                dcc.Store('session-store', storage_type='session'),
                create_navbar(),
                dmc.AppShellMain(
                    dmc.Container(
                        [
                            # TODO only use spinnger on home screen. It does not seem to work for other screen.
                            app_spinner_container := html.Div(
                                loading_page_spinner := dash_loading_spinners.Pacman(
                                    fullscreen=True,
                                    id='loading_page_spinner'
                                ),
                                id='app_spinner_container'
                            ),

                            page_content_container := dmc.Container(
                                dash.page_container,
                                id='page_content_container'
                            )
                        ],
                        style={'border': '5px dashed red'}
                    ),
                )
            ],
            header={'height': 50}
        )
    )
)


@callback(
    Output(app_spinner_container, 'children'),
    Input(page_content_container, 'loading_state'),
    State(app_spinner_container, 'children'),
)
def hide_page_loading_spinner(
    _,
    children,
):
    print("hide_page_loading_spinner")
    if children:
        return None
    raise PreventUpdate
