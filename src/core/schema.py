import json
from enum import Enum
from dataclasses import dataclass, field, asdict

from omegaconf import MISSING

class DataType(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"

#### Hydra Config Schema ####
@dataclass
class DataLoaderConfig:
    pass

@dataclass
class DatasetConfig:
    name: str = MISSING
    n_classes: int = MISSING
    raw_data: dict = MISSING
    human_adapter: dict | None = None
    data_type: DataType = MISSING

@dataclass 
class ModelConfig:
    pass

@dataclass
class QueryStrategyConfig:
    pass

""" @dataclass
class ActiveMlConfig:
    random_seed: int = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    query_strategy: QueryStrategyConfig = field(default_factory=QueryStrategyConfig) """

@dataclass
class ActiveMlConfig:
    random_seed: int = MISSING
    model: ModelConfig | None = MISSING
    dataset: DatasetConfig = MISSING
    query_strategy: QueryStrategyConfig = MISSING

#### Session Config #### 
@dataclass
class SessionConfig:
    n_cycles: int = 10
    batch_size: int = 10
    max_candidates: int | float = 1000

@dataclass
class SessionState:
    n_labeled: int = 0


#### Batch State ####
@dataclass
class Batch:
    indices: list[int]
    progress: int # progress
    annotations: list[int]

    def to_json(self):
        return json.dumps(asdict(self))
    
    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return Batch(**data)