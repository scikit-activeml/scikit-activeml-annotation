import logging
from pathlib import Path
from functools import lru_cache
from typing import TypeVar

import omegaconf
from omegaconf import OmegaConf

import pydantic

import skactiveml_annotation.paths as sap

T = TypeVar("T", bound=pydantic.BaseModel)

# TODO: Remove string from this signature
@lru_cache(maxsize=5)
def parse_yaml_config_dir(dir_path: Path | str, clazz: type[T]) -> list[T]:
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    yaml_files_paths = (f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() == ".yaml")
    return [parse_yaml_file(file_path, clazz) for file_path in yaml_files_paths]

def parse_yaml_file(file_path: Path | str, clazz: type[T]) -> T:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file() or file_path.suffix != ".yaml":
        msg = f"Cannot find yaml file at path: {file_path}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    try:
        # Add another field id which is the filename stem so it can be used
        # later when composing configuration to check which model etc was
        # selected by the user
        file_id = file_path.stem
        cfg = OmegaConf.load(file_path)
        OmegaConf.update(cfg, "id", file_id, merge=True)
        cfg_raw = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        logging.error(
            "\n".join([
                f"Failed to parse yaml file: {file_path}",
                f"Error: {e}",
            ])
        )
        raise

    try:
        return clazz.model_validate(cfg_raw)
    except pydantic.ValidationError as e:
        logging.error(
            "\n".join([
                f"Pydantic validation for {clazz.__name__} failed for config at {file_path}.",
                f"Error: {e}",
            ])
        )
        raise

# TODO: overrides should never be None here
def overrides_to_list(overrides: tuple[tuple[str, str], ...]) -> list[str]:
    # eg. [("dataset","mnist"), …] → ["dataset=mnist", …]
    return [f'{group}={name}' for group, name in overrides]


def set_ids_from_overrides(cfg: omegaconf.DictConfig, overrides: tuple[tuple[str, str], ...]):
    """
    Uses the provided override values to set the add 'id' field for each config category.
    To enable checking which option (config yaml file) was selected

    Expected format of each override tuple:
      (key, override_value)
    The '+' prefix is stripped if present.
    """
    for key, override_value in overrides:
        key = key.lstrip('+')

        # Set the id attribute of that config node.
        cfg_id = f"{key}.id"

        # Allows to add a key.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, cfg_id, override_value, merge=True)
        OmegaConf.set_struct(cfg, True)


def is_dataset_cfg_overridden(dataset_id: str) -> bool:
    path = sap.OVERRIDE_CONFIG_DATASET_PATH / f'{dataset_id}.yaml'
    return path.exists()
