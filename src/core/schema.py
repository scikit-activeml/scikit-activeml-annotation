import json
from enum import Enum
from dataclasses import dataclass, field, asdict

import numpy as np
from omegaconf import MISSING

from embedding.base import EmbeddingBaseAdapter


class DataType(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"


# region Hydra Config Schema
@dataclass
class EmbeddingConfig:
    id: str = MISSING
    display_name: str = MISSING
    # TODO definition has wrong type here.
    definition: EmbeddingBaseAdapter = MISSING


@dataclass
class DatasetConfig:
    id: str = MISSING
    display_name: str = MISSING  # Name that will be displayed in ui for that dataset
    classes: list[str] = MISSING  # All the possible data labels.
    data_path: str = MISSING  # Path to data dir. Relative to project root
    data_type: DataType = MISSING


@dataclass
class ModelConfig:
    id: str = MISSING
    display_name: str = MISSING
    definition: dict = MISSING


@dataclass
class QueryStrategyConfig:
    id: str = MISSING
    display_name: str = MISSING
    # TODO This could be automated
    model_agnostic: bool = MISSING
    definition: str = MISSING


@dataclass
class ActiveMlConfig:
    random_seed: int = MISSING
    model: ModelConfig | None = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    query_strategy: QueryStrategyConfig = field(default_factory=QueryStrategyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
# endregion


# region Session Config
@dataclass
class SessionConfig:
    batch_size: int = 10  # How many samples to label before retraining
    subsampling: int | float = 1000  #


# endregion

# region Batch State
@dataclass
class Batch:
    # TODO use correct datatypes
    indices: list[int]  # TODO these might have to be renamved to embedding_indices.
    annotations: list[int]
    class_probas: np.ndarray  # shape len(indices) x num_of_classes
    progress: int  # progress

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return Batch(**data)

    def is_completed(self) -> bool:
        return self.progress >= len(self.indices)


@dataclass
class Annotation:
    embedding_idx: int
    file_name: str
    label: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return Annotation(**data)
# endregion
