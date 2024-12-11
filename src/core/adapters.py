
from abc import ABC, abstractmethod

class DataLoaderAdapter(ABC):

    @abstractmethod
    def get_raw_data():
        pass
