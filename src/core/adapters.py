from abc import ABC, abstractmethod
from typing import Callable
from pathlib import Path

from hydra.utils import instantiate

from core.schema import DataLoaderConfig

from sklearn.utils import Bunch
import numpy as np

class DataLoaderAdapter(ABC):
    @abstractmethod
    def get_raw_data(idx: int):
        raise NotImplementedError
    
    @abstractmethod
    def get_human_data(idx: int):
        raise NotImplementedError
    
    @abstractmethod
    def get_all_label_names() -> list[str]:
        raise NotImplementedError
    

class SklearnImageDataAdapter(DataLoaderAdapter):
    def __init__(
            self,
            dataloader: DataLoaderConfig, 
            cache: bool = False,
            cache_path: Path = None, 
        ):
        super().__init__()

        print("Instantiate data_loader")
        bunch: Bunch = instantiate(dataloader)

        self.data = bunch.data
        # TODO normalize these images?
        self.images = bunch.images
        self.label_names = self._init_label_names(bunch)


    def get_raw_data(self, idx: int) -> np.ndarray:
        # TODO this is not used.
        return self.data[idx]

    def get_human_data(self, idx: int):
        return self.images[idx]

    def get_all_label_names(self) -> list[str]:
        """
        Returns all label names available in the dataset.
        """
        return self.label_names

    def _init_label_names(self, bunch: Bunch) -> list[str]:
        if 'target_names' in bunch:
            label_names = bunch.target_names

        # Some datasets dont return target_names
        label_names = np.unique(bunch.target)
        return label_names
    

""" class SklearnTextDataAdapter(DataLoaderAdapter):
    def __init__(
            self,
            data_loader: Callable, 
            cache: bool = False,
        ):
        super().__init__() """
    
