# Std
from dataclasses import dataclass, field
from typing import Callable
from functools import partial

# 3rd Party
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from numpy import ndarray

from skactiveml.utils import call_func, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from skactiveml.base import QueryStrategy

import numpy as np

# own
from util.path import (ROOT_PATH, CONFIG_PATH)
from backend.session import (SessionConfig, SessionState, fetch_session_config)
from backend.service import (finish_label_session, req_annotation)

# tmp
from inspect import signature

# These variable will come from the gui:

# TODO Use this to tell each config. What fields it has to have.
@dataclass
class DataLoaderConfig:
    pass

@dataclass
class DatasetConfig:
    n_classes: int = MISSING

@dataclass 
class ModelConfig:
    pass

@dataclass
class QueryStrategyConfig:
    pass

@dataclass
class ActiveMlConfig:
    random_seed: int = MISSING
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    query_strategy: QueryStrategyConfig = MISSING

cs = ConfigStore().instance()
cs.store(name='base_config', node=ActiveMlConfig)

def load_data(cfg: ActiveMlConfig) -> tuple[ndarray, ndarray]:
    X, Y = instantiate(cfg.dataset.definition)
    return X, Y


def build_classifier(
        model_cfg: ModelConfig, 
        dataset_cfg: DatasetConfig,
        random_state: np.random.RandomState
    ) -> SklearnClassifier:
    n_classes = dataset_cfg.n_classes

    model_package_name = str(model_cfg.definition._target_).split(".")[0]
    if "skactiveml" == model_package_name:
        return instantiate(model_cfg.definition, random_state=random_state, classes=np.arange(n_classes))
        
    else:
        clf = instantiate(model_cfg.definition)
        return SklearnClassifier(clf, random_state=random_state, classes=np.arange(n_classes))
          

def create_regressor(cfg: ModelConfig, random_state: np.random.RandomState):
    raise NotImplementedError


""" def start_active_learning_cycle(qs: ):
    for _ in range(n_cycles): """


def filter_kwargs(func: Callable, **kwargs) -> Callable:
    params = signature(func).parameters
    param_names = params.keys()

    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_kwargs:
        # If the func accepts **kwargs, no filtering is needed
        return partial(func, **kwargs)

    # Otherwise, filter only the kwargs that match func's signature
    filtered_kwargs = {p_name: p_obj for p_name, p_obj in kwargs.items() if p_name in param_names}   

    return partial(func, **filtered_kwargs)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name='config')
def build_activeMl_pipeline(cfg: ActiveMlConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    # Allow partially labeled datasets?
    X, y_true = load_data(cfg)
    y = np.full(len(X), MISSING_LABEL)

    random_state = np.random.RandomState(cfg.random_seed)
    session_config: SessionConfig = fetch_session_config()

    clf: SklearnClassifier = build_classifier(cfg.model, cfg.dataset, random_state=random_state)

    # max_candidates for subsampling.
    qs: QueryStrategy = instantiate(cfg.query_strategy.definition, random_state=random_state)
    qs: SubSamplingWrapper = SubSamplingWrapper(qs, max_candidates=session_config.max_candidates, random_state=random_state)

    # Don't fit classifier in query function. To avoid fitting twice.
    query_func: Callable = filter_kwargs(qs.query, batch_size=session_config.batch_size, clf=clf, fit_clf=False, discriminator=clf)
    
    c = 0
    while c < session_config.n_cycles:
        query_indices = query_func(X=X, y=y)
        print(query_indices)

        # blocking call
        y_labeled = req_annotation(query_indices, cfg.dataset.n_classes, random_state)
        y[query_indices] = y_labeled

        clf.fit(X=X, y=y)
        c += 1

if __name__ == '__main__':
    build_activeMl_pipeline()