from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

import skactiveml_annotation.paths as sap
from skactiveml_annotation.core.shared_types import DashProgressFunc


def relative_to_root(path: Path) -> Path:
    """
    Convert an absolute path to a path that is relative to the project root.
    """
    return path.relative_to(sap.ROOT_PATH)


class EmbeddingBaseAdapter(ABC):
    @abstractmethod
    def compute_embeddings(self, data_path: Path, progress_func: DashProgressFunc) -> tuple[np.ndarray, list[Path]]:
        """
        Compute and return the feature matrix and corresponding file paths for the given directory of data.

        This function loads the data from the specified directory, preprocesses it,
        and potentially creates embeddings such that the resulting feature matrix X
        has shape (num_samples, num_features). The following invariant must hold:

        - X[i] corresponds to file_paths[i] for all i.

        Args:
            data_path (str): The absolute path to the directory containing the data
                             to be processed and embedded.

        Returns:
            tuple: A tuple containing:
                - `np.ndarray`: The feature matrix of shape (num_samples, num_features).
                - `list[str]`: A list of file paths relative to the root of the project.
                                Used to display human readable sample.
        """
        pass

