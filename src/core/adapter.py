from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from util.path import DATASETS_PATH
from core.schema import DatasetConfig
from util.path import CACHE_PATH


class BaseAdapter(ABC):
    @abstractmethod
    def compute_embeddings(self, data_path: Path) -> tuple[np.ndarray, list[str]]:
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
                - `list[str]`: A list of file paths corresponding to the samples in `X`.
        """
        pass

    # @abstractmethod
    # def get_supported_datatypes(self):
    #     pass

    def get_or_compute_embeddings(
            self,
            dataset_cfg: DatasetConfig,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Resolve the data_path path, check/load cache if enabled, call compute_features,
        and cache the result if needed.
        """
        dataset_id = dataset_cfg.id
        data_path = dataset_cfg.data_path

        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = DATASETS_PATH / data_path

        # Unique key
        cache_key = f"{dataset_id}_{self.__class__.__name__}"
        print(f"cache key: {cache_key}")

        cache_path = Path(str(CACHE_PATH)) / f"{cache_key}.npz"  # Use .npz to store multiple arrays

        if cache_path.exists():
            print(f"Cache hit. Loading cached features from {cache_path}")
            # Load both the feature matrix and the file names from the .npz cache
            with np.load(str(cache_path)) as data:
                X = data['X']
                file_names = data['file_names'].tolist()  # Convert to a list if necessary
            return X, file_names

        print("Cache miss. Computing feature matrix and caching ...")
        X, file_names = self.compute_embeddings(data_path)

        # Cache both the feature matrix and the file names in the .npz file
        np.savez(str(cache_path), X=X, file_names=file_names)

        return X, file_names


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, transform: callable):
        self.transform = transform
        self.image_paths = list(data_path.glob('*'))  # This will list all files in the directory
        self.file_names = [path.name for path in self.image_paths]  # Get file_names only

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        file_name = self.file_names[idx]

        try:
            # Open the image using Pillow
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, file_name  # If there's an error, return None and the file name

        image = self.transform(image)  # Apply the transformations
        return image, file_name


class TorchVisionAdapter(BaseAdapter):
    def __init__(self,
                 batch_size: int = 16,
                 model_variant: str = "dinov2_vitb14"):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(26),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        try:
            # Load the pretrained model from PyTorch Hub
            self.model = torch.hub.load("facebookresearch/dinov2", model_variant)
        except Exception as e:
            raise RuntimeError("DINOv2 model is not available") from e

        # Move the model to the appropriate device (GPU or CPU)
        self.model = self.model.to(self.device)
        # Set the model to evaluation mode
        self.model.eval()

    def compute_embeddings(self, data_path: Path) -> tuple:
        """
        Load images from the directory in batches, process them through the model,
        and return the concatenated feature matrix and corresponding file names.
        """
        dataset = ImageDataset(data_path, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        embeddings_list = []
        file_names_list = []

        # Disable gradient tracking since we are in inference mode
        with torch.no_grad():
            for batch, file_names in dataloader:
                assert len(batch) == len(file_names)

                # Filter out any failed image loads (None entries)
                batch = [img for img in batch if img is not None]
                if not batch:
                    continue

                # Stack the images into a tensor and move to the correct device
                batch_tensor = torch.stack(batch).to(self.device)

                # Get embeddings for all images in the batch
                embeddings = self.model(batch_tensor)

                # Append the embeddings and corresponding file names
                embeddings_list.append(embeddings.cpu().numpy())
                file_names_list.extend(file_names)  # Ensure file_names match embeddings

        # Concatenate all embeddings into a single feature matrix
        feature_matrix = np.concatenate(embeddings_list, axis=0)

        # Return the feature matrix and corresponding file_names
        return feature_matrix, file_names_list


class SimpleFlattenAdapter(BaseAdapter):
    def __init__(self):
        pass

    def compute_embeddings(self, data_path: Path) -> tuple[np.ndarray, list[str]]:
        """
        Load images one by one from the directory, flatten them,
        and return the stacked feature matrix.
        """
        feature_list = []
        # iterdir does not ensure order of files in dir.
        files = [str(file) for file in data_path.iterdir() if file.is_file()]

        for file in files:
            try:
                # img = Image.open(file).convert("RGB")
                img = Image.open(file)
                if img.mode == "L":
                    # Greyscale image
                    img_data = np.array(img)  # Shape (H, W)
                else:
                    # RGB image
                    img_data = np.array(img.convert("RGB"))  # Shape (H, W, 3)

                feature = img_data.flatten().reshape(1, -1)
                feature_list.append(feature)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        try:
            feature_matrix = np.concatenate(feature_list, axis=0)
        except Exception as e:
            raise RuntimeError(f"Some images are RBG while others are Greyscale: {e}")

        return feature_matrix, files
