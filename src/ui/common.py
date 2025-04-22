from core.schema import ActiveMlConfig
from util.deserialize import compose_config
from ui.storekey import StoreKey


def compose_from_state(store_data) -> ActiveMlConfig:
    overrides = (
        ('dataset', store_data[StoreKey.DATASET_SELECTION.value]),
        ('query_strategy', store_data[StoreKey.QUERY_SELECTION.value]),
        ('embedding', store_data[StoreKey.EMBEDDING_SELECTION.value]),
        ('+model', store_data[StoreKey.MODEL_SELECTION.value])  # add model to default list
    )

    return compose_config(overrides)
