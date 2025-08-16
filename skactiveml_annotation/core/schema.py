# pyright: reportAny=false
# pyright: reportExplicitAny=false
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
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

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any) -> DataType:
        return _instantiate(self, DataType, **kwargs)


class QueryStrategyTarget(pydantic.BaseModel):
    target_: str = Field(..., alias="_target_")

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any):
        return _instantiate(self, SingleAnnotatorPoolQueryStrategy , **kwargs)

class ModelTarget(pydantic.BaseModel):
    target_: str = Field(..., alias="_target_")

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any):
        return _instantiate(self, ClassifierMixin, **kwargs)

class EmbeddingTarget(pydantic.BaseModel):
    target_: str = Field(..., alias="_target_")

    class Config:
        extra: str = "allow"

    def instantiate(self, **kwargs: Any):
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
    model: ModelConfig | None = None
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
    subsampling: str | int | float | None = None

    def __post_init__(self):
        if self.subsampling == '':
            self.subsampling = None


@dataclass
class Batch:
    # TODO use correct datatypes
    emb_indices: list[int]
    annotations: list[str]
    class_probas: list[list[float]]  # shape len(indices) x num_of_classes
    progress: int  # progress
    # Meta data
    start_times: list[str] = field(default_factory=list)
    end_times: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return Batch(**data)

    def is_completed(self) -> bool:
        return self.progress >= len(self.emb_indices)


@dataclass
class Annotation:
    embedding_idx: int
    file_name: str
    label: str
    start_time: str = ''
    end_time: str = ''

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return Annotation(**data)


@dataclass
class AutomatedAnnotation:
    embedding_idx: int
    file_name: str
    label: int
    confidence: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return AutomatedAnnotation(**data)
