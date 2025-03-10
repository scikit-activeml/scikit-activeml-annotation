from pathlib import Path
from torchvision.datasets import CIFAR10
from util.path import DATASETS_PATH


def save_cifar10_images(output_dir=DATASETS_PATH / 'cifar10_images'):
    dataset = CIFAR10(root=str(DATASETS_PATH / 'cifar10'), download=True, train=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each image as a PNG file. Each entry is a tuple (image, label)
    for idx, (image, label) in enumerate(dataset):
        # The image is already a PIL Image (RGB, 0-255), so no additional processing is needed.
        image.save(output_dir / f"cifar10_{idx}.png")

    print(f"Saved {len(dataset)} images in '{output_dir}'")


if __name__ == "__main__":
    save_cifar10_images()
