from pathlib import Path

from torchvision.datasets import (
    VisionDataset,
    CIFAR10,
    MNIST
)

from paths import DATASETS_PATH


def load_cifar10_images():
    name = 'cifar10'
    base_path = DATASETS_PATH / f"{name}"
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
    base_path = DATASETS_PATH / f"{name}"
    _load_torchvision_images(
        name,
        base_path,
        MNIST(
            root=base_path / 'temp',
            download=True,
            train=True
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

    print(f"Saved {len(dataset)} images in '{output_dir}'")


if __name__ == "__main__":
    load_mnist_images()
    # save_cifar10_images()
