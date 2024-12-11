import json
from enum import Enum
from dataclasses import dataclass, field, asdict

from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from typing import Dict

class DataType(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"

#### Hydra Config Schema ####


""" @dataclass
class DatasetConfig:
    name: str = MISSING
    n_classes: int = MISSING
    raw_data: dict = MISSING
    human_adapter: dict | None = None
    data_type: DataType = MISSING """

@dataclass
class DataLoaderConfig:
    _target_: str = MISSING

@dataclass
class AdapterConfig:
    _target_: str = MISSING
    dataloader: DataLoaderConfig = MISSING

@dataclass
class DatasetConfig:
    name: str = MISSING
    data_type: DataType = MISSING
    n_classes: int = MISSING
    # adapter_cfg: Dict = MISSING
    adapter_cfg: AdapterConfig = MISSING
    # adapter: DataLoaderAdapter = None
    # adapter: DataLoaderAdapter = field(init=False) # Deferred initialization

    """ raw_data: dict = MISSING
    human_adapter: dict | None = None """

    def __post_init__(self):
        # print("POST INIT IS INVOKED")
        """ if not isinstance(self.adaptor_cfg, DictConfig):
            raise ValueError """
        # instantiate the adaptor from cfg

        """ print(type(self.adapter_cfg))
        self.adapter = instantiate(self.adapter_cfg) """
        # self.adapter = instantiate(self.adapter_cfg)
        """ print(self.adapter_cfg)
        print(type(self.adapter_cfg))
        self.adapter = instantiate(self.adapter_cfg) """

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

""" @dataclass
class ActiveMlConfig:
    random_seed: int = MISSING
    model: ModelConfig | None = MISSING
    dataset: DatasetConfig = MISSING
    query_strategy: QueryStrategyConfig = MISSING """

#### Session Config #### 
@dataclass
class SessionConfig:
    n_cycles: int = 10
    batch_size: int = 10
    max_candidates: int | float = 1000


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