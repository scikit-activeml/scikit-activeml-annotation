from enum import Enum, auto


class StoreKey(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name  # Automatically use the name of the member as its value

    DATASET_SELECTION = auto()
    EMBEDDING_SELECTION = auto()
    QUERY_SELECTION = auto()
    MODEL_SELECTION = auto()
    BATCH_STATE = auto()


class AnnotProgress(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    PROGRESS = auto()
    TOTAL_NUM = auto()


class DataDisplayCfgKey(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    RESCALE_FACTOR = auto()
    RESAMPLING_METHOD = auto()
