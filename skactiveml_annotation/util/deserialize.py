from pathlib import Path
from functools import lru_cache

import omegaconf
from omegaconf import OmegaConf

import skactiveml_annotation.paths as sap

@lru_cache(maxsize=5)
def parse_yaml_config_dir(dir_path: Path | str) -> list[omegaconf.DictConfig]:
    """
    Parses YAML config files in a directory and returns a dictionary mapping
    file names (without extension) to their DictConfig objects.

    Args:
        dir_path (Path | str): Path to the directory containing YAML files.

    Returns:
        list[DictConfig]: a List of DictConfig's
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    configs = []
    for path in dir_path.iterdir():
        if path.is_file() and path.suffix.lower() == '.yaml':
            try:
                config = OmegaConf.load(path)
                # Use file name as id
                file_id = path.stem
                OmegaConf.update(config, "id", file_id, merge=True)
                configs.append(config)
            except Exception as e:
                print(f"Failed to parse YAML file {path}: {e}")

    return configs


def parse_yaml_file(file_path: Path | str) -> omegaconf.DictConfig | None:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file() or file_path.suffix.lower() != ".yaml":
        print(f"file_path: {file_path} is not a path to a yaml file!")
        return None

    try:
        file_id = file_path.stem
        cfg = OmegaConf.load(file_path)
        OmegaConf.update(cfg, "id", file_id, merge=True)
        return OmegaConf.load(file_path)
    except Exception as e:
        print(f"Failed to parse YAML file {file_path}: {e}")


def overrides_to_list(overrides: tuple[tuple[str, str], ...]) -> list[str]:
    # eg. [("dataset","mnist"), …] → ["dataset=mnist", …]
    if overrides is None:
        return []
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


def is_dataset_cfg_overridden(dataset_id) -> bool:
    path = sap.OVERRIDE_CONFIG_DATASET_PATH / f'{dataset_id}.yaml'
    return path.exists()
