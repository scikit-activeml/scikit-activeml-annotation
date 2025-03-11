from inspect import signature
from functools import partial
from typing import Callable

import numpy as np
from numpy import ndarray

from skactiveml.utils import call_func, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from skactiveml.base import QueryStrategy

from core.schema import *
from core.adapter import *

from util.deserialize import parse_yaml_config_dir
from util.path import DATA_CONFIG_PATH, ANNOTATED_PATH, QS_CONFIG_PATH, DATASETS_PATH, MODEL_CONFIG_PATH

MISSING_LABEL_JSON = "Missing"


def _load_data_raw(cfg: ActiveMlConfig) -> ndarray:
    print("LOADING DATA")
    X, _ = instantiate(cfg.dataset.raw_data)
    print("FINISH LOADING DATA")
    print(type(X))
    return X


def _build_classifier(
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        random_state: np.random.RandomState
) -> SklearnClassifier:
    n_classes = len(dataset_cfg.label_names)
    model_package_name = str(model_cfg.definition._target_).split(".")[0]
    if "skactiveml" == model_package_name:
        return instantiate(model_cfg.definition, random_state=random_state, classes=np.arange(n_classes))

    else:
        clf = instantiate(model_cfg.definition)
        return SklearnClassifier(clf, random_state=random_state, classes=np.arange(n_classes))


def _create_regressor(cfg: ModelConfig, random_state: np.random.RandomState):
    raise NotImplementedError


# TODO can use from skactiveml.utils import call_func instead?
def _filter_kwargs(func: Callable, **kwargs) -> Callable:
    params = signature(func).parameters
    param_names = params.keys()

    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_kwargs:
        # If the func accepts **kwargs, no filtering is needed
        return partial(func, **kwargs)

    # Otherwise, filter only the kwargs that match func's signature
    filtered_kwargs = {p_name: p_obj for p_name, p_obj in kwargs.items() if p_name in param_names}

    # print("filtered_kwargs", filtered_kwargs)

    return partial(func, **filtered_kwargs)


# TODO always load dataset over and over again.
def _setup_query(cfg: ActiveMlConfig, session_cfg: SessionConfig) -> Callable:
    random_state = np.random.RandomState(cfg.random_seed)

    clf: SklearnClassifier = _build_classifier(cfg.model, cfg.dataset, random_state=random_state)

    # max_candidates for subsampling.
    qs: QueryStrategy = instantiate(cfg.query_strategy.definition, random_state=random_state)
    qs: SubSamplingWrapper = SubSamplingWrapper(qs, max_candidates=session_cfg.max_candidates,
                                                random_state=random_state)

    # TODO seperate query from fitting?
    query_func: Callable = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=clf, fit_clf=True,
                                          discriminator=clf)
    return query_func
    """ query_indices = query_func(X=X, y=y)
    return partial(query_func, X=X, y=y) """


# region API
def get_dataset_config_options() -> dict[str, DatasetConfig]:
    config_options: dict[str, DatasetConfig] = parse_yaml_config_dir(DATA_CONFIG_PATH)
    return config_options


def get_qs_config_options() -> dict[str, QueryStrategyConfig]:
    return parse_yaml_config_dir(QS_CONFIG_PATH)


def get_model_config_options() -> dict[str, ModelConfig]:
    return parse_yaml_config_dir(MODEL_CONFIG_PATH)


def get_human_readable_sample(dataset_cfg: DatasetConfig, idx: int):
    """Allow the UI to request a human-readable sample.
       Assumption: Each file in the directory represents one sample (an image).
    """
    data_type = dataset_cfg.data_type
    if data_type in (DataType.AUDIO, DataType.TEXT):
        raise NotImplementedError("Human readable sample for AUDIO or TEXT is not implemented.")

    path = dataset_cfg.data_path

    # Ensure the path is absolute using DATASETS_PATH if necessary.
    if not Path(path).is_absolute():
        path = DATASETS_PATH / path

    # List and sort files in the directory for a deterministic order.
    files = sorted([
        str(file)
        for file in Path(path).iterdir()
        if file.is_file()
    ])

    if idx < 0 or idx >= len(files):
        raise IndexError(f"Index {idx} is out of bounds for dataset with {len(files)} files.")

    sample_file = files[idx]
    # Load and return the image.
    sample_image = Image.open(sample_file).convert("RGB")
    return sample_image


def get_all_label_options():
    raise NotImplementedError


def request_query(cfg: ActiveMlConfig,
                  session_cfg: SessionConfig,
                  adapter: BaseAdapter) -> np.ndarray:

    X = adapter.process_directory(cfg.dataset)
    labels = _load_or_init_annotations(X, cfg.dataset.id)

    query_func = _setup_query(cfg, session_cfg)
    print("Querying the active ML model ...")
    # If the query function expects an array, create one based on ordered indices.
    labels_list = [labels.get(str(idx), MISSING_LABEL) for idx in range(len(X))]
    query_indices = query_func(X=X, y=np.array(labels_list))
    return query_indices


def _load_or_init_annotations(X: np.ndarray, dataset_id: str) -> dict:
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'

    if json_file_path.exists():
        with json_file_path.open('r') as f:
            labels = json.load(f)
    else:
        # No annotations yet initialize new JSON File putting labels to missing.
        labels = {str(idx): MISSING_LABEL for idx in range(len(X))}
        with json_file_path.open("w") as f:
            json.dump(labels, f, indent=4)
    return labels


def completed_batch(dataset_id: str, batch: Batch):
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'

    print("completed batch")
    print(json_file_path)

    if not json_file_path.exists():
        raise RuntimeError("JSON file should already exist here!")

    with json_file_path.open('r') as f:
        labels = json.load(f)

    # Update labeled data with new annotations.
    # Use string keys to avoid duplicates.
    for idx, annotation in zip(batch.indices, batch.annotations):
        labels[str(idx)] = annotation

    # Write back the updated labels (this overwrites the existing file).
    with json_file_path.open("w") as f:
        json.dump(labels, f, indent=4)


# endregion


def finish_label_session():
    # Notify UI that. There is not more labels to label.
    raise NotImplementedError


def stop_labelling_session():
    raise NotImplementedError


def retrain():
    raise NotImplementedError
