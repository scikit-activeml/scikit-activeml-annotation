import sys
import logging
from pathlib import Path
import random

try:
    # Optional dependency
    from datasets import load_dataset  # pyright: ignore[reportAttributeAccessIssue]
except ImportError as e:
    # TODO: Add better error msg
    logging.error(e)
    sys.exit()

# Use paths logic from package
package_root = Path(__file__).resolve().parent.parent
sys.path.append(str(package_root))

import skactiveml_annotation.paths as sap

# Set a fixed seed for reproducible shuffling
SEED = 42
random.seed(SEED)


def load_imdb_texts():
    """Load IMDb reviews and save each review as a separate file."""
    name = 'imdb'
    base_path = sap.DATASETS_PATH / "raw_text" / name
    _load_hf_texts(name, base_path, dataset_name="imdb", split='train', text_field='text')


def load_ag_news_texts():
    """Load AG News dataset (unlabeled)."""
    name = 'ag_news'
    base_path = sap.DATASETS_PATH / "raw_text" / name
    _load_hf_texts(name, base_path, dataset_name="ag_news", split='train', text_field='text')

def load_sms_spam_texts():
    """Load SMS Spam Collection dataset (unlabeled)."""
    name = 'sms_spam'
    base_path = sap.DATASETS_PATH / "raw_text" / name
    _load_hf_texts(
        name,
        base_path,
        dataset_name="sms_spam",
        split='train',
        text_field='sms',
        label_field='label'
    )


def _load_hf_texts(name: str, path: Path, dataset_name: str, split: str, text_field: str, label_field: str | None = None):
    """
    Helper function to download and save Hugging Face text datasets
    Each sample is saved as a separate .txt file in a single folder.
    """
    output_dir = path / f"{name}_texts"
 
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split)
    
    texts = list(dataset[text_field]) 

    # TODO must I shuffle here?
    # random.shuffle(texts)

    for idx, text in enumerate(texts):
        file_path = output_dir / f"{name}_{idx}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    if label_field is not None:
        output_dir_labels = path / f"{name}_labels"
        output_dir_labels.mkdir(parents=True, exist_ok=True)
        labels = list(dataset[label_field])

        for idx, label in enumerate(labels):
            file_path_label = output_dir_labels / f"{name}_{idx}.txt"

            with open(file_path_label, "w", encoding="utf-8") as f:
                f.write(str(label))

    logging.info(f"Saved {len(texts)} samples in '{output_dir}'")


if __name__ == "__main__":
    # load_imdb_texts()
    # load_ag_news_texts()
    load_sms_spam_texts()
    pass
