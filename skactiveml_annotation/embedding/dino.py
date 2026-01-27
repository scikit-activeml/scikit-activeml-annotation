from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from skactiveml_annotation.util import logging

try:
    import torch # pyright: ignore[reportMissingImports]
    from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]
    import torchvision.transforms as transforms # pyright: ignore[reportMissingImports]
except ImportError as e:
    logging.error(e)
    raise

from skactiveml_annotation.core.shared_types import DashProgressFunc

from .base import (
    relative_to_root,
    EmbeddingBaseAdapter
)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, transform: Callable):
        self.transform = transform
        self.image_paths = self._collect_valid_images(data_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _collect_valid_images(self, data_path: Path) -> list[Path]:
            """
            Iterate over files in data_path and keep only valid images.
            Invalid or unreadable images are skipped with a warning.
            """
            logging.info("Collecting valid images ...")

            valid_paths: list[Path] = []

            for path in data_path.iterdir():
                if not path.is_file():
                    continue

                try:
                    with Image.open(path) as img:
                        img.verify()  # integrity check
                    valid_paths.append(path)
                except Exception as e:
                    logging.warning(f"Skipping invalid image file: {path} ({e})")

            return valid_paths

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]

        try:
            # Open the image using Pillow
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # This error should not occure becauce all image files are checked
            # beforehand.
            logging.error(f"Unexpected error loading image {image_path}: {e}")
            raise

        # Transform pil image to a tensor of shape (3, h, w), containg raw data
        image_tensor: torch.Tensor = self.transform(image)
        return image_tensor, str(relative_to_root(image_path))


class TorchVisionAdapter(EmbeddingBaseAdapter):
    def __init__(self,
                 batch_size: int = 16,
                 model_variant: str = "dinov2_vitb14"):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        try:
            # Load the pretrained model from PyTorch Hub
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_variant)
        except Exception as e:
            raise RuntimeError("DINOv2 model is not available") from e

        # Move the model to the appropriate device (GPU or CPU)
        self.model = self.model.to(self.device)
        # Set the model to evaluation mode
        self.model.eval()


    def compute_embeddings(
        self,
        data_path: Path,
        progress_func: DashProgressFunc
    ) -> tuple[np.ndarray, list[Path]]:
        """
        Load images from the directory in batches, process them through the model,
        and return the concatenated feature matrix and corresponding file names.
        """
        logging.info(f"Compute Torchvision embedding using device: {self.device}")

        dataset = ImageDataset(data_path, self.transform)
        n_samples = len(dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        embeddings_list = []
        file_path_list = []

        processed_samples = 0
        steps = 100
        next_report = steps

        # Disable gradient tracking since we are in inference mode
        with torch.no_grad():
            for batch_tensor, file_paths in dataloader:
                # Move the batch_tensor to the correct device
                batch_tensor = batch_tensor.to(self.device)

                # Get embeddings for all images in the batch
                embeddings = self.model(batch_tensor)

                # Append the embeddings and corresponding file names
                embeddings_list.append(embeddings.cpu().numpy())
                file_path_list.extend(file_paths)  # Ensure file_paths match embeddings

                # Update progress counter and print every 1000 samples
                processed_samples += batch_tensor.shape[0]
                if processed_samples >= next_report:
                    next_report += steps
                    progress_func((processed_samples / n_samples) * 100)

        # Concatenate all embeddings into a single feature matrix
        feature_matrix = np.concatenate(embeddings_list, axis=0)

        # Return the feature matrix and corresponding file_paths
        return feature_matrix, [Path(s) for s in file_path_list]
