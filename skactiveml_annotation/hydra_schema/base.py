from enum import Enum

import pydantic

from ._internal import (
    EmbeddingTarget,
    ModelTarget,
    QueryStrategyTarget
)


class Modality(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"

# all the ids are derived from the file name and must not be part of hydra config files
class DatasetConfig(pydantic.BaseModel):
    id: str
    display_name: str
    classes: list[str]
    data_path: str
    modality: Modality

class EmbeddingConfig(pydantic.BaseModel):
    id: str
    display_name: str
    definition: EmbeddingTarget
    modalities: list[Modality]

class QueryStrategyConfig(pydantic.BaseModel):
    id: str
    display_name: str
    model_agnostic: bool
    definition: QueryStrategyTarget

class ModelConfig(pydantic.BaseModel):
    id: str
    display_name: str
    definition: ModelTarget

class ActiveMlConfig(pydantic.BaseModel):
    random_seed: int
    model: ModelConfig
    dataset: DatasetConfig
    query_strategy: QueryStrategyConfig
    embedding: EmbeddingConfig
