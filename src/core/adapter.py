from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from util.path import DATASETS_PATH
from core.schema import DatasetConfig
from util.path import CACHE_PATH


class BaseAdapter(ABC):
    @abstractmethod
    def compute_features(self, directory: str) -> np.ndarray:
        """
        Compute and return the feature matrix for the given directory.
        The directory is provided as an absolute path.
        """
        pass

    def process_directory(self, dataset_cfg: DatasetConfig, cache_dir: str = str(CACHE_PATH)) -> np.ndarray:
        """
        Resolve the directory path, check/load cache if enabled, call compute_features,
        and cache the result if needed.
        """
        dataset_id = dataset_cfg.id
        directory = dataset_cfg.data_path

        # Convert to an absolute path if necessary.
        directory = Path(directory)
        if not directory.is_absolute():
            directory = DATASETS_PATH / directory

        # Create a unique cache key based on dataset id and the class name.
        cache_key = f"{dataset_id}_{self.__class__.__name__}"
        print(f"cache key: {cache_key}")

        cache_path = Path(cache_dir) / f"{cache_key}.npy"

        if cache_path.exists():
            print(f"Cache hit. Loading cached features from {cache_path}")
            return np.load(str(cache_path))

        print("Cache miss. Computing feature matrix and caching ...")
        X = self.compute_features(str(directory))
        np.save(str(cache_path), X)
        return X


class ImageDataset(Dataset):
    def __init__(self, directory: str, transform=None):
        self.directory = Path(directory)
        self.transform = transform
        self.files = [file for file in self.directory.iterdir() if file.is_file()]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        try:
            img = Image.open(file).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None


# --- Concrete Implementation Using DINOv2 via torch.hub (with batching) ---
class TorchVisionAdapter(BaseAdapter):
    def __init__(self,
                 batch_size: int = 16,
                 model_variant: str = "dinov2_vitb14"):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_variant, pretrained=True)
            print(f"Using DINOv2 model variant: {model_variant}")
        except Exception as e:
            raise RuntimeError(
                "DINOv2 model not available. Please update your torch.hub or torchvision version."
            ) from e

        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_features(self, directory: str) -> np.ndarray:
        """
        Load images from the directory in batches, process them through the model,
        and return the concatenated feature matrix.
        """
        dataset = ImageDataset(directory, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        features_list = []
        for batch in dataloader:
            # Filter out any failed image loads.
            batch = [img for img in batch if img is not None]
            if not batch:
                continue
            batch_tensor = torch.stack(batch).to(self.device)
            with torch.no_grad():
                output = self.model(batch_tensor)
                if output.dim() == 4:
                    output = F.adaptive_avg_pool2d(output, (1, 1))
                    output = output.view(output.size(0), -1)
            features_list.append(output.cpu().numpy())
        return np.concatenate(features_list, axis=0)


# --- Concrete Implementation Without Batching ---
class SimpleFlattenAdapter(BaseAdapter):
    def __init__(self):
        self.transform = None  # No transform is applied by default.

    def compute_features(self, directory: str) -> np.ndarray:
        """
        Load images one by one from the directory, flatten them,
        and return the stacked feature matrix.
        """
        feature_list = []
        files = [str(file) for file in Path(directory).iterdir() if file.is_file()]
        for file in files:
            try:
                # TODO why convert to RGB?
                # img = Image.open(file).convert("RGB")
                img = Image.open(file)
                if self.transform:
                    img = self.transform(img)
                else:
                    img = np.array(img)
                # Flatten the image and add an extra dimension.
                feature = img.flatten()[None, :]
                feature_list.append(feature)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        return np.concatenate(feature_list, axis=0)
