from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class DataLoaderConfig:
    pass

@dataclass
class DatasetConfig:
    name: str = MISSING
    n_classes: int = MISSING

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
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    query_strategy: QueryStrategyConfig = MISSING