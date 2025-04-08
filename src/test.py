import dash
from dash import (
    Dash,
    Input,
    Output,
    State,
    html,
    dcc,
    callback,
    DiskcacheManager
)
import diskcache
from dash.exceptions import PreventUpdate

import dash_mantine_components as dmc
import dash_loading_spinners

from ui.components.navbar import create_navbar
from paths import (
    PAGES_PATH,
    ASSETS_PATH
)

app = Dash(
    __name__,
    external_stylesheets=[dmc.theme.DEFAULT_THEME, dmc.styles.ALL],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
)


app.layout = dmc.MantineProvider(
    dmc.Center(
        [
            dmc.Button(id='trigger', style={'display': 'None'}),
            # dcc.Store(id='trigger'),
            dmc.Stack(
                [
                    dmc.Button("button", id='button'),
                    dmc.Button('no update button', id='no-update')
                ]
            )
        ],
        h='100vh'
    )
)


@callback(
    Output('trigger', 'n_clicks', allow_duplicate=True),
    Input('no-update', 'n_clicks'),
    prevent_initial_call=True,
)
def no_update(
    _
):
    return dash.no_update


@callback(
    Output('trigger', 'n_clicks', allow_duplicate=True),
    Input('button', 'n_clicks'),
    prevent_initial_call=True
)
def button_click(
    clicks,
):
    return None


@callback(
    Input('trigger', 'n_clicks'),
    prevent_initial_call=True,
)
def on_trigger(
    _
):
    print("callback has been triggered!")


def main():
    app.run(debug=True, port=9090)


if __name__ == '__main__':
    main()
