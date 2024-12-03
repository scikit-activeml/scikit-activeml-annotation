from inspect import signature
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Callable, Generator

import numpy as np
from numpy import ndarray

from hydra.utils import instantiate, call
from omegaconf import OmegaConf

from skactiveml.utils import call_func, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from skactiveml.base import QueryStrategy

from .schema import *
from util.deserialize import compose_config

@dataclass
class SessionConfig:
    n_cycles: int = 10
    batch_size: int = 1
    max_candidates: int | float = 1000

@dataclass
class SessionState:
    n_labeled: int = 0


def fetch_session_config() -> SessionConfig:
    # TODO actually request data from UI.
    # for now return dummy data
    return SessionConfig()


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


def get_label_names():
    pass

def _build_activeMl_pipeline(cfg: ActiveMlConfig):
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

    def pipeline():
        c = 0
        while c < session_config.n_cycles:
            query_indices = query_func(X=X, y=y)
            print(query_indices)

            # blocking call
            # y_labeled = req_annotation(query_indices, cfg.dataset.n_classes, random_state)
            y_labeled = yield query_indices[0]
            y[query_indices] = y_labeled

            clf.fit(X=X, y=y)
            c += 1

    return pipeline

def setup_activeMl_cycle(session_cfg: SessionConfig, overrides: Dict[str, str] | None = None) -> Generator:
    cfg: ActiveMlConfig = compose_config(overrides)
    return _build_activeMl_pipeline(cfg)


def stop_labelling_session():
    raise NotImplementedError


def req_annotation(query_indices: np.ndarray, n_classes: int, random_state: np.random.RandomState) -> np.ndarray:
    """
    Returns np array-like of shape (query_indices)
    """

    # TODO for now select random classes for each requested label.
    return random_state.randint(0, n_classes, size=len(query_indices))


def finish_label_session():
    # Notify UI that. There is not more labels to label.
    print("Not Implemented")


def retrain():
    raise NotImplementedError