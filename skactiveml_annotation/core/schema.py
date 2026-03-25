from datetime import datetime, timedelta
import json
from typing import Self, override
from dataclasses import dataclass

import isodate
import pydantic

MISSING_LABEL_MARKER = 'MISSING_LABEL'
DISCARD_MARKER = 'DISCARDED'


@dataclass
class SessionConfig:
    batch_size: int = 10  # How many samples to label before retraining
    subsampling: int | float | None = None

    def __post_init__(self):
        # Workarround Dash initialized it to emtpy str?
        if self.subsampling == '':
            self.subsampling = None


class AnnotationMetaData(pydantic.BaseModel):
    first_view_time: datetime      # Time when the sample was first presented
    last_edit_time: datetime       # Last time when a change was made
    total_view_duration: timedelta # Total presentation time
    skip_intended_cnt: int = 0     # How many time the sample has been activly skipped

    @pydantic.field_validator("total_view_duration", mode="before")
    @classmethod
    def parse_duration(cls, value: str | timedelta) -> timedelta:
        if isinstance(value, str):
            return isodate.parse_duration(value)
        return value

    @pydantic.field_serializer("total_view_duration")
    def serialize_duration(self, value: timedelta) -> str:
        return isodate.duration_isoformat(value)


class Annotation(pydantic.BaseModel):
    embedding_idx: int
    label: str
    meta_data: AnnotationMetaData



class Batch(pydantic.BaseModel):
    progress: int = 0
    emb_indices: list[int]
    classes_sklearn: list[str]
    annotations: list[Annotation | None] = []
    class_probas: list[list[float]] | None = None

    _min_progress: int = pydantic.PrivateAttr()
    _max_progress: int = pydantic.PrivateAttr()

    def init(self) -> Self:
        if not (0 <= self.progress <= len(self.emb_indices)):
            raise ValueError("Initial progress out of range")
        self._min_progress = self.progress
        self._max_progress = self.progress

        return self


    def get_annotation_not_none(self) -> Annotation:
        annot = self.get_annotation()
        if annot is None:
            # TODO:
            raise ValueError
        return annot


    def get_emb_index(self) -> int:
        return self.emb_indices[self.progress]

    def get_annotation(self) -> Annotation | None:
        return self.annotations[self.progress]

    def add_annotation(self, annot: Annotation):
        self.annotations[self.progress] = annot

    def get_progress_percent(self) -> float:
        return (self.progress / len(self.emb_indices)) * 100
    
    def concat(self, other: Self, progress: int = 0) -> Self:
        class_probas = (
            self.class_probas + other.class_probas
            if self.class_probas is not None and other.class_probas is not None
            else None
        )

        return type(self)(
            progress=progress,
            emb_indices=self.emb_indices + other.emb_indices,
            # TODO: why use only other?
            classes_sklearn=other.classes_sklearn,
            class_probas=class_probas,
            annotations=self.annotations + other.annotations,
        ).init()

    # Have to override pydantic behaviour because it does not serde
    # private attributes
    @override
    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        data["_min_progress"] = self._min_progress
        data["_max_progress"] = self._max_progress
        return data

    @override
    @classmethod
    def model_validate(cls, data: dict | Self, *args, **kwargs) -> Self:
        if isinstance(data, cls):
            return data

        data = dict(data)
        _min_progress = data.pop("_min_progress")
        _max_progress = data.pop("_max_progress")

        batch = super().model_validate(data, *args, **kwargs)

        batch._min_progress = _min_progress
        batch._max_progress = _max_progress
        return batch


    @override
    def model_dump_json(self, *args, **kwargs) -> str:
        return json.dumps(self.model_dump(*args, mode='json', **kwargs))

    @override
    @classmethod
    def model_validate_json(cls, data: str, *args, **kwargs) -> Self:
        return cls.model_validate(json.loads(data))

    def advance(self, step: int):
        if not self.is_advanceable(step):
            raise ValueError(
                f"Cannot advance batch by {step} because it would result out of bounds"
            )

        self.progress += step
        self._min_progress = min(self._min_progress, self.progress)
        self._max_progress = max(self._max_progress, self.progress)

    def _is_valid_progress(self, progress: int) -> bool:
        return 0 <= progress < len(self.emb_indices)

    def get_num_annotated(self) -> int:
        return self._max_progress - self._min_progress + 1

    def is_advanceable(self, step: int) -> bool:
        next_progress = self.progress + step
        return self._is_valid_progress(next_progress)

    def is_completed(self) -> bool:
        return not self._is_valid_progress(self.progress)

    def __len__(self) -> int:
        return len(self.emb_indices)


class HistoryIdx(pydantic.BaseModel):
    idx: int


class AnnotationProgress(pydantic.BaseModel):
    num_annotated: int
    num_samples: int

    def is_all_annotated(self) -> bool:
        return self.num_annotated == self.num_samples

class AutomatedAnnotation(pydantic.BaseModel):
    embedding_idx: int
    label: str
    confidence: float
