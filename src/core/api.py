import pickle
from inspect import signature
from functools import partial
from typing import Callable

from numpy import ndarray

from skactiveml.utils import call_func, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from skactiveml.base import QueryStrategy

from core.schema import *
from core.adapter import *

from util.deserialize import parse_yaml_config_dir
from util.path import DATA_CONFIG_PATH, ANNOTATED_PATH, QS_CONFIG_PATH, DATASETS_PATH


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


def _filter_kwargs(func: Callable, **kwargs) -> Callable:
    params = signature(func).parameters
    param_names = params.keys()

    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_kwargs:
        # If the func accepts **kwargs, no filtering is needed
        return partial(func, **kwargs)

    # Otherwise, filter only the kwargs that match func's signature
    filtered_kwargs = {p_name: p_obj for p_name, p_obj in kwargs.items() if p_name in param_names}

    return partial(func, **filtered_kwargs)


# TODO always load dataset over and over again.
def _setup_query(cfg: ActiveMlConfig, session_cfg: SessionConfig) -> Callable:
    random_state = np.random.RandomState(cfg.random_seed)

    clf: SklearnClassifier = _build_classifier(cfg.model, cfg.dataset, random_state=random_state)

    # max_candidates for subsampling.
    qs: QueryStrategy = instantiate(cfg.query_strategy.definition, random_state=random_state)
    qs: SubSamplingWrapper = SubSamplingWrapper(qs, max_candidates=session_cfg.max_candidates,
                                                random_state=random_state)

    # Don't fit classifier in query function. To avoid fitting twice.
    query_func: Callable = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=clf, fit_clf=False,
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


def load_label_data(dataset_name: str):
    pickle_file_path = ANNOTATED_PATH / f'{dataset_name}.pkl'

    if pickle_file_path.exists():
        with pickle_file_path.open('rb') as f:
            labels = pickle.load(f)
            return labels


def request_query(
        cfg: ActiveMlConfig,
        session_cfg: SessionConfig,
        adapter: BaseAdapter
) -> np.ndarray:
    pickle_file_path = ANNOTATED_PATH / f'{cfg.dataset.id}.pkl'

    X = adapter.process_directory(cfg.dataset)

    # X = adapter.get_raw_data()
    # X = _load_data_raw(cfg)
    print(type(X))
    # print("SHAPE")
    print(X.shape)

    if pickle_file_path.exists():
        with pickle_file_path.open('rb') as f:
            labels: dict[int, int] = pickle.load(f)
    else:
        # Initialize pickle file.
        labels = np.full(len(X), MISSING_LABEL)
        with pickle_file_path.open("wb") as f:
            pickle.dump(labels, f)

    query_func = _setup_query(cfg, session_cfg)
    query_indices = query_func(X=X, y=labels)
    return query_indices


def completed_batch(dataset_name: str, batch: Batch):
    pickle_file_path = ANNOTATED_PATH / f'{dataset_name}.pkl'

    if not pickle_file_path.exists():
        raise RuntimeError("Pickle file should allready exist here!")

    with pickle_file_path.open('rb') as f:
        labels = pickle.load(f)

    # Update labeled data with new annotations
    labels[batch.indices] = batch.annotations

    # Write back updated labels
    with pickle_file_path.open("wb") as f:
        pickle.dump(labels, f)


# endregion


def finish_label_session():
    # Notify UI that. There is not more labels to label.
    raise NotImplementedError


def stop_labelling_session():
    raise NotImplementedError


def retrain():
    raise NotImplementedError
