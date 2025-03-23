from typing import Dict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from dataclasses import dataclass, field

import hydra
from hydra import initialize, compose
from hydra.utils import instantiate, call
from hydra.core.config_store import ConfigStore
from hydra.initialize import initialize_config_dir

from omegaconf import DictConfig, OmegaConf, MISSING

from .path import CONFIG_PATH
from core.schema import ActiveMlConfig


def _dict_overrides_to_list(overrides: Dict[str, str]) -> list[str]:
    out = []
    for idx, (key, value) in enumerate(overrides.items()):
        if value is not None:
            out.append(f'{key}={value}')

    return out


def compose_config(overrides: Dict[str, str] | None = None) -> ActiveMlConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        # TODO
        # schema: DictConfig = OmegaConf.structured(ActiveMlConfig)

        if overrides is not None:
            overrides = _dict_overrides_to_list(overrides)

        return compose('config', overrides=overrides)

        # TODO validation is no longer possible because model could be null
        # # Make sure cfg has at least the attributes that schema has.
        # try:
        #     # TODO how do get validation right without losing fields?
        #     cfg: ActiveMlConfig = OmegaConf.merge(cfg, schema)
        #
        #     # Allow additional fields
        #     OmegaConf.set_struct(cfg, False)
        #
        #     """ cfg: DictConfig = OmegaConf.merge(cfg, schema)
        #     OmegaConf.resolve(cfg)
        #     cfg: ActiveMlConfig = OmegaConf.to_object(cfg)
        #     print("Converted ActiveMlConfig")
        #     print(OmegaConf.to_yaml(cfg)) """
        #     return cfg
        # except Exception as e:
        #     print(f'Validation of Schema failed because: {e}')
        #     exit(-1)
######################


def parse_yaml_config_dir(dir_path: Path | str) -> dict[str, DictConfig]:
    """
    Parses YAML config files in a directory and returns a dictionary mapping
    file names (without extension) to their DictConfig objects.

    Args:
        dir_path (Path | str): Path to the directory containing YAML files.

    Returns:
        dict[str, DictConfig]: A dictionary where keys are config file names
                               (without .yaml) and values are DictConfig objects.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    configs = {}
    for path in dir_path.iterdir():
        if path.is_file() and path.suffix.lower() == '.yaml':
            try:
                config_name = path.stem  # Get file name without extension
                configs[config_name] = OmegaConf.load(path)
            except Exception as e:
                print(f"Failed to parse YAML file {path}: {e}")

    return configs


def parse_yaml_file(file_path: Path | str) -> DictConfig | None:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file() or file_path.suffix.lower() != ".yaml":
        print(f"file_path: {file_path} is not a path to a yaml file!")
        return None

    try:
        return OmegaConf.load(file_path)
    except Exception as e:
        print(f"Failed to parse YAML file {file_path}: {e}")
