import sys
import logging
from pathlib import Path
import random

try:
    # Optional dependency
    from datasets import load_dataset  # pyright: ignore[reportAttributeAccessIssue]
except ImportError as e:
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
        text_field='sms'
    )


def _load_hf_texts(name: str, path: Path, dataset_name: str, split: str, text_field: str):
    """
    Helper function to download and save Hugging Face text datasets
    Each sample is saved as a separate .txt file in a single folder.
    """
    output_dir = path / f"{name}_texts"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split)
    
    print(dataset)

    texts = list(dataset[text_field]) 

    random.shuffle(texts)

    for idx, text in enumerate(texts):
        file_path = output_dir / f"{name}_{idx}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Saved {len(texts)} samples in '{output_dir}'")


if __name__ == "__main__":
    # Example usage
    # load_imdb_texts()
    # load_ag_news_texts()
    load_sms_spam_texts()
    pass
