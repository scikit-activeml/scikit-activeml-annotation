import json
from enum import Enum
from dataclasses import dataclass, field, asdict

from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig


class DataType(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"


# region Hydra Config Schema
@dataclass
class DataLoaderConfig:
    _target_: str = MISSING


@dataclass
class AdapterConfig:
    _target_: str = MISSING
    dataloader: DataLoaderConfig = MISSING


@dataclass
class DatasetConfig:
    # name: str = MISSING
    display_name: str = MISSING  # Name that will be displayed in ui for that dataset
    label_names: list[str] = MISSING  # All the possible data labels.
    data_path: str = MISSING  # Path to data dir. Path has to be Absolute or relative to datasets dir.

    data_type: DataType = MISSING
    # adapter_cfg: AdapterConfig = MISSING


@dataclass
class ModelConfig:
    pass


@dataclass
class QueryStrategyConfig:
    pass


@dataclass
class ActiveMlConfig:
    random_seed: int = MISSING
    model: ModelConfig | None = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    query_strategy: QueryStrategyConfig = field(default_factory=QueryStrategyConfig)


# endregion

# region Session Config
@dataclass
class SessionConfig:
    batch_size: int = 10  # How many samples to label before retraining
    max_candidates: int | float = 1000  #


# endregion

# region Batch State
@dataclass
class Batch:
    indices: list[int]
    progress: int  # progress
    annotations: list[int]

    def to_json(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return Batch(**data)
# endregion
