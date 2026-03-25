

from enum import IntEnum, auto

import pydantic

from skactiveml_annotation.core import api

class SelectionStep(IntEnum):
    @staticmethod
    def _generate_next_value_(name: str, start, count, last_values):
        # Make it start at 0 for some reason it does not.
        return count

    DATASET = auto()
    EMBEDDING = auto()
    QUERY = auto()
    MODEL = auto()
    SAMPLING_PARAM = auto()


class Selection(pydantic.BaseModel):
    dataset_id: str
    embedding_id: str
    query_id: str
    model_id: str

    @classmethod
    def size(cls) -> int:
        return len(cls.model_fields)


class SelectionProgress(pydantic.BaseModel):
    selections: list[str | None] = [None] * Selection.size()

    def add(self, step: SelectionStep, val: str | None):
        self.selections[step.value] = val

    def convert(self) -> Selection:
        selections = list(filter(api.not_none_type_narrowing, self.selections))

        if len(selections) < Selection.size():
            missing = [
                field_name
                for i, field_name in enumerate(Selection.model_fields.keys())
                if self.selections[i] is None
            ]
            raise ValueError(f"Missing selections for: {', '.join(missing)}")

        return Selection(**dict(zip(Selection.model_fields.keys(), selections)))
        
    def get(self, key: SelectionStep) -> str | None:
        return self.selections[key.value]

    def get_not_none(self, key: SelectionStep) -> str:
        val = self.selections[key.value]
        if val is None:
            raise ValueError(f"Selection for step '{key.name}' has not been made yet")
        return val
