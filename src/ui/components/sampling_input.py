
import dash_mantine_components as dmc


def create_sampling_inputs():
    return [
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
            thousandSeparator=' ',
        ),

        # Subsampling selection
        dmc.NumberInput(
            label="Subsampling",
            id='subsampling-input',
            allowNegative=False,
            debounce=True,
            hideControls=True,
            thousandSeparator=' ',
            persistence='subsampling-persistence',
            persistence_type='local',
        ),
    ]
