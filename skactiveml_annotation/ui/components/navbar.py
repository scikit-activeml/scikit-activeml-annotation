import dash_mantine_components as dmc

def create_navbar(**kwargs: object):
    _ = kwargs

    return dmc.AppShellHeader(
        [
            dmc.Flex(
                [
                    dmc.Anchor(
                        children='Home',
                        href='/',
                        size="xl",
                        style={
                            'color': "var(--mantine-color-white)"
                        },
                    ),
                ],
                justify='flex-start',
                direction='row',
                align='center',
                style={
                    "marginLeft": "3vw",
                    'height': '100%'
                }
            )
        ],
        style={'backgroundColor': "var(--mantine-color-dark-7)"},
        # width={"base": 250},  # Adjust the width as needed
        # p="md",
        withBorder=True
    )



