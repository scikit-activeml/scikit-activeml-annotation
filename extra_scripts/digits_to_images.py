import os
from sklearn import datasets
from PIL import Image
import numpy as np

from util.path import DATASETS_PATH


def save_digits_images(output_dir=str(DATASETS_PATH / 'digits_images')):
    # Load digits dataset
    digits = datasets.load_digits()
    images = digits.images

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each image as a PNG file
    for idx, image in enumerate(images):
        img = preprocess_image(image)
        img.save(os.path.join(output_dir, f"digit_{idx}.png"))

    print(f"Saved {len(images)} images in '{output_dir}'")


def preprocess_image(image: np.ndarray) -> Image.Image:
    """
    Normalize an image array to the 0-255 range and convert it to an 8-bit PIL Image.

    This function handles images that are not already in the 0-255 range (e.g., images from
    the sklearn digits dataset with a maximum value of 16). If the image values are greater than 1,
    the image is scaled so that its maximum value becomes 255. The resulting array is then converted
    to an unsigned 8-bit integer format, which is the expected format for PIL.

    Parameters:
        image (np.ndarray): The input image array.

    Returns:
        Image.Image: A PIL Image object with pixel values in the 0-255 range.
    """
    # If the image values exceed 1, assume the image is not normalized and scale it
    if image.max() > 1:
        # Scale the image so that the maximum value becomes 255
        image = 255 * (image / image.max())

    image = image.astype(np.uint8)

    pil_image = Image.fromarray(image)
    return pil_image


if __name__ == "__main__":
    save_digits_images()
