import sys
import logging
from pathlib import Path

try:
    # Optional dependency
    from torchvision.datasets import (  # pyright: ignore[reportMissingImports]
        VisionDataset,
        CIFAR100,
        CIFAR10,
        MNIST,
        STL10,
        FashionMNIST
    )
except ImportError as e:
    # TODO: Add better error msg
    logging.error(e)
    sys.exit()


# Use paths logic from package
package_root = Path(__file__).resolve().parent.parent
sys.path.append(str(package_root))

import skactiveml_annotation.paths as sap


def load_cifar100_images():
    name = 'cifar100'
    base_path = sap.DATASETS_PATH / f"{name}"
    _load_torchvision_images(
        name,
        base_path,
        CIFAR100(
            root=base_path / 'temp',
            download=True,
            train=True,
        )
    )


def load_cifar10_images():
    name = 'cifar10'
    base_path = sap.DATASETS_PATH / f"{name}"
    _load_torchvision_images(
        name,
        base_path,
        CIFAR10(
            root=base_path / "temp",
            download=True,
            train=True
        )
    )


def load_mnist_images():
    name = 'mnist'
    base_path = sap.DATASETS_PATH / f"{name}"
    _load_torchvision_images(
        name,
        base_path,
        MNIST(
            root=base_path / 'temp',
            download=True,
            train=True
        )
    )


def load_fashion_mnist():
    name = 'fashion-mnist'
    base_path = sap.DATASETS_PATH / f"{name}"
    _load_torchvision_images(
        name,
        base_path,
        FashionMNIST(
            root=base_path / 'temp',
            download=True,
            train=True,
        )
    )


def load_stl10_images():
    name = 'stl10'
    base_path = sap.DATASETS_PATH / f"{name}"
    _load_torchvision_images(
        name,
        base_path,
        STL10(
            root=base_path / 'temp',
            download=True,
            split='unlabeled',
        )
    )


def _load_torchvision_images(
    name: str,
    path: Path,
    dataset: VisionDataset
):
    output_dir = path / f"{name}_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each image as a PNG file. Each entry is a tuple (image, label)
    for idx, (image, label) in enumerate(dataset):
        image.save(output_dir / f'{name}_{idx}.png')

    logging.info(f"Saved {len(dataset)} images in '{output_dir}'")


if __name__ == "__main__":
    # load_cifar100_images()
    # load_cifar10_images()
    # load_mnist_images()
    # load_fashion_mnist()
    # load_stl10_images()
    pass
