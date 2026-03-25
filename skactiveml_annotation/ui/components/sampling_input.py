
import dash_mantine_components as dmc

from skactiveml_annotation.ui.pages.annotation import ids

def create_sampling_inputs():
    return [
        # Batch Size selection
        dmc.NumberInput(
            label="Batch Size",
            id=ids.BATCH_SIZE_INPUT,
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
            id=ids.SUBSAMPLING_INPUT,
            allowNegative=False,
            debounce=True,
            hideControls=True,
            thousandSeparator=' ',
            persistence='subsampling-persistence',
            persistence_type='local',
        ),
    ]
