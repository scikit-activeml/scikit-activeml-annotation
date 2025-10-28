import json
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from typing import Any, Literal, TypeVar 

import hydra

import pydantic
from pydantic import Field

from sklearn.base import ClassifierMixin
from skactiveml.base import SingleAnnotatorPoolQueryStrategy

from skactiveml_annotation.embedding.base import EmbeddingBaseAdapter

MISSING_LABEL_MARKER = 'MISSING_LABEL'
DISCARD_MARKER = 'DISCARDED'

DataTypeLiteral = Literal["skactiveml_annotation.core.schema.DataType"] 

T = TypeVar("T")

class DataType(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"

class DataTypeTarget(pydantic.BaseModel):
    # ... tells pydantic this field is needed
    target_: DataTypeLiteral = Field(..., alias="_target_")
    args_: list[str] = Field(..., alias="_args_")

    # Tell pydantic to allow extra keys
    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any) -> DataType:
        return _instantiate(self, DataType, **kwargs)


class QueryStrategyTarget(pydantic.BaseModel):
    target_: str = Field(..., alias="_target_")

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any) -> SingleAnnotatorPoolQueryStrategy:
        return _instantiate(self, SingleAnnotatorPoolQueryStrategy, **kwargs)

class ModelTarget(pydantic.BaseModel):
    target_: str = Field(..., alias="_target_")

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any) -> ClassifierMixin:
        return _instantiate(self, ClassifierMixin, **kwargs)

class EmbeddingTarget(pydantic.BaseModel):
    target_: str = Field(..., alias="_target_")

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any) -> EmbeddingBaseAdapter:
        return _instantiate(self, EmbeddingBaseAdapter, **kwargs)

class EmbeddingConfig(pydantic.BaseModel):
    id: str
    display_name: str
    definition: EmbeddingTarget 


class DatasetConfig(pydantic.BaseModel):
    id: str
    display_name: str
    classes: list[str]
    data_path: str
    data_type: DataTypeTarget


class ModelConfig(pydantic.BaseModel):
    id: str
    display_name: str
    definition: ModelTarget


class QueryStrategyConfig(pydantic.BaseModel):
    id: str
    display_name: str
    model_agnostic: bool
    definition: QueryStrategyTarget


class ActiveMlConfig(pydantic.BaseModel):
    random_seed: int
    model: ModelConfig
    dataset: DatasetConfig
    query_strategy: QueryStrategyConfig
    embedding: EmbeddingConfig


def _instantiate(cfg: pydantic.BaseModel, expected_type: type[T], **kwargs: Any) -> T:
    try:
        cfg_dict = cfg.model_dump(by_alias=True)
        x = hydra.utils.instantiate(cfg_dict, **kwargs)
        # TODO: instantiate can fail
    except Exception as e:
        logging.error(
            "\n".join([
                f"Hydra failed to instantiate instance of: {expected_type.__name__}.",
                f"Config: {cfg.model_dump(by_alias=True)}",
                f"Exception: {e}",
            ])
        )
        raise

    if not isinstance(x, expected_type):
        logging.error("\n".join([
            "Hydra instantiated unexpected type:",
            f"Expected type: {expected_type.__name__}",
            f"Actual type:   {type(x).__name__}",
        ]))
        raise TypeError(
            f"Expected instance of {expected_type.__name__}, got {type(x).__name__}"
        )
  
    return x


@dataclass
class SessionConfig:
    batch_size: int = 10  # How many samples to label before retraining
    subsampling: int | float | None = None

    def __post_init__(self):
        # Workarround Dash initialized it to emtpy str?
        if self.subsampling == '':
            self.subsampling = None


class Batch:
    def __init__(
        self, 
        emb_indices: list[int], 
        classes_sklearn: list[str],
        class_probas: list[list[float]] | None = None, 
        progress: int = 0
    ):
        if not (0 <= progress <= len(emb_indices)):
            raise ValueError("Initial progress out of range")

        self.emb_indices = emb_indices
        self.class_probas = class_probas
        self.classes_sklearn = classes_sklearn

        self._progress = progress
        self._min_progress = progress
        self._max_progress = progress

    @property
    def progress(self) -> int:
        return self._progress

    # TODO maybe its cleaner if this returns a boolean is completed?
    def advance(self, step: int):
        self._progress += step

        if self.is_completed():
            return

        self._min_progress = min(self._min_progress, self._progress)
        self._max_progress = max(self._max_progress, self._progress)

    def get_num_annotated(self) -> int:
        return self._max_progress - self._min_progress + 1

    def is_completed(self) -> bool:
        return self.progress < 0 or self.progress >= len(self.emb_indices)

    def __len__(self) -> int:
        return len(self.emb_indices)

    # -- Serialization & Deserialization --
    def to_json(self) -> str:
        data = {
            "emb_indices": self.emb_indices,
            "class_probas": self.class_probas,
            "classes_sklearn": self.classes_sklearn,
            "_progress": self._progress,
            "_min_progress": self._min_progress,
            "_max_progress": self._max_progress
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "Batch":
        data = json.loads(json_str)
        batch = cls(
            emb_indices=data["emb_indices"],
            class_probas=data.get("class_probas", None),
            classes_sklearn=data.get("classes_sklearn", None),
            progress=data["_progress"]
        )
        batch._min_progress = data["_min_progress"]
        batch._max_progress = data["_max_progress"]
        return batch


class Annotation(pydantic.BaseModel):
    embedding_idx: int
    label: str

    first_view_time: str = ''
    total_view_duration: str = ''
    last_edit_time: str = ''


class AnnotationList(pydantic.BaseModel):
    annotations: list[Annotation | None]

class HistoryIdx(pydantic.BaseModel):
    idx: int

@dataclass
class AutomatedAnnotation:
    embedding_idx: int
    file_path: str
    label: int
    confidence: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return AutomatedAnnotation(**data)

