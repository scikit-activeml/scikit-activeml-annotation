from typing import Dict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

###### 

import logging
from dataclasses import dataclass, field

import hydra
from hydra import initialize, compose
from hydra.utils import instantiate, call
from hydra.core.config_store import ConfigStore
from hydra.initialize import initialize_config_dir

from omegaconf import DictConfig, OmegaConf, MISSING

from util.path import CONFIG_PATH
from core.schema import ActiveMlConfig;

logging.basicConfig(format='[%(levelname)s]: %(message)s')


def _dict_overrides_to_list(overrides: Dict[str, str]) -> list[str]:
    out = [None] * len(overrides)
    for idx, (key, value) in enumerate(overrides.items()):
        out[idx] = f'{key}={value}'

    return out    

def compose_config() -> ActiveMlConfig:
    return compose_config(None)

def compose_config(overrides: Dict[str, str] | None) -> ActiveMlConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        # TODO here we can override default config.yaml

        if not overrides is None: 
            overrides = _dict_overrides_to_list(overrides)
        cfg: DictConfig = compose('config', overrides=overrides)

        schema: ActiveMlConfig = OmegaConf.structured(ActiveMlConfig)

        # Make sure cfg has at least the attributes that schema has.
        try:
            # TODO how do get validation right without losing fields?
            cfg: ActiveMlConfig = OmegaConf.merge(cfg, schema)
            """ cfg: DictConfig = OmegaConf.merge(cfg, schema)
            OmegaConf.resolve(cfg)
            cfg: ActiveMlConfig = OmegaConf.to_object(cfg)
            print("Converted ActiveMlConfig")
            print(OmegaConf.to_yaml(cfg)) """
            return cfg
        except Exception as e:
            logging.error(f'Validation of Schema failed because: {e}')
            exit(-1)

######################

def parse_yaml_config_dir(dir_path: Path | str) -> list[DictConfig]:
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    out = []
    for path in dir_path.iterdir():
        if path.is_file() and path.suffix.lower() == '.yaml':
            try:
                out.append(OmegaConf.load(path))
            except Exception as e:
                print(f"Failed to parse YAML file {path}: {e}")

    return out

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
