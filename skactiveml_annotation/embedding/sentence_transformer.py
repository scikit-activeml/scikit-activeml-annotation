import logging
from pathlib import Path

import numpy as np

try:
    import sentence_transformers  # pyright: ignore[reportMissingImports]
except ImportError as e:
    logging.error(e)
    raise

from skactiveml_annotation.core.shared_types import DashProgressFunc

from .base import (
    relative_to_root,
    EmbeddingBaseAdapter
)

# TODO possibly create template class for this?
class SentenceTransformerAdapter(EmbeddingBaseAdapter):
    def __init__(
        self,
        model_variant: str
    ):
        self.model_variant = model_variant

    def compute_embeddings(
        self, 
        data_path: Path,
        progress_func: DashProgressFunc
    ) -> tuple[np.ndarray, list[Path]]:
        _ = progress_func

        # 1. Validate input directory
        if not data_path.exists() or not data_path.is_dir():
            raise ValueError(f"Provided path {data_path} is not a valid directory.")

        text_files_paths = sorted(data_path.glob("*.txt"))

        if not text_files_paths:
            raise ValueError(f"No .txt files found in {data_path}")

        # 3. Read content of each file
        samples = [path.read_text(encoding="utf-8") for path in text_files_paths]
        file_paths = [relative_to_root(path) for path in text_files_paths]

        model = sentence_transformers.SentenceTransformer(self.model_variant)

        # 4. Compute embeddings
        embeddings = model.encode(samples, normalize_embeddings=True)

        logging.info("Embedding complete with shape: %s", embeddings.shape)

        return embeddings, file_paths
