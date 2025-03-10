from enum import Enum, auto


class StoreKey(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name  # Automatically use the name of the member as its value

    SELECTIONS = auto()
    BATCH_STATE = auto()

