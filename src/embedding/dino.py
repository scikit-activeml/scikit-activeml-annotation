from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .base import (
    relative_to_root,
    EmbeddingBaseAdapter
)

from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, transform: callable):
        self.transform = transform
        self.image_paths = [path for path in data_path.iterdir() if path.is_file()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            # Open the image using Pillow
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, image_path  # If there's an error, return None and the file name

        image = self.transform(image)  # Apply the transformations
        return image, relative_to_root(image_path)


class TorchVisionAdapter(EmbeddingBaseAdapter):
    def __init__(self,
                 batch_size: int = 16,
                 model_variant: str = "dinov2_vitb14"):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define image transformations
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

    def compute_embeddings(self, data_path: Path, progress_func=None) -> tuple:
        """
        Load images from the directory in batches, process them through the model,
        and return the concatenated feature matrix and corresponding file names.
        """
        print(f"Compute Torchvision embedding using device: {self.device}")
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
            for batch, file_paths in dataloader:

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
                file_path_list.extend(file_paths)  # Ensure file_paths match embeddings

                # Update progress counter and print every 1000 samples
                processed_samples += len(batch)
                if processed_samples >= next_report:
                    print(f"Processed {processed_samples} samples")
                    next_report += steps
                    progress_func((processed_samples / n_samples) * 100)

        # Concatenate all embeddings into a single feature matrix
        feature_matrix = np.concatenate(embeddings_list, axis=0)

        # Return the feature matrix and corresponding file_paths
        return feature_matrix, file_path_list