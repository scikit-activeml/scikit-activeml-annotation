from skactiveml_annotation.core.api import compose_config
from skactiveml_annotation.core.schema import ActiveMlConfig
from skactiveml_annotation.ui.storekey import StoreKey

def compose_from_state(store_data) -> ActiveMlConfig:
    overrides = (
        ('dataset', store_data[StoreKey.DATASET_SELECTION.value]),
        ('query_strategy', store_data[StoreKey.QUERY_SELECTION.value]),
        ('embedding', store_data[StoreKey.EMBEDDING_SELECTION.value]),
        ('+model', store_data[StoreKey.MODEL_SELECTION.value])  # add model to default list
    )

    return compose_config(overrides)
