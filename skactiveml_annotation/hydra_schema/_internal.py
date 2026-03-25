import logging
from typing import (
    Any,
    TypeVar,
)

import hydra

import pydantic
from pydantic import Field

from skactiveml.base import (
    ClassifierMixin,
    SingleAnnotatorPoolQueryStrategy,
)
from skactiveml_annotation.embedding.base import EmbeddingBaseAdapter

T = TypeVar("T")


class QueryStrategyTarget(pydantic.BaseModel):
    target_: str = pydantic.Field(..., alias="_target_")

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


def _instantiate(cfg: pydantic.BaseModel, expected_type: type[T], **kwargs: Any) -> T:
    try:
        cfg_dict = cfg.model_dump(by_alias=True)
        x = hydra.utils.instantiate(cfg_dict, **kwargs)
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
