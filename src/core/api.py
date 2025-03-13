from inspect import signature
from functools import partial
from typing import Callable

from hydra.utils import instantiate

from skactiveml.utils import MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from skactiveml.base import QueryStrategy

from core.schema import *
from core.adapter import *

from util.deserialize import parse_yaml_config_dir
from util.path import DATA_CONFIG_PATH, ANNOTATED_PATH, QS_CONFIG_PATH, MODEL_CONFIG_PATH


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


# TODO can use from skactiveml.utils import call_func instead?
def _filter_kwargs(func: Callable, **kwargs) -> Callable:
    params = signature(func).parameters
    param_names = params.keys()

    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_kwargs:
        # If the func accepts **kwargs, no filtering is needed
        return partial(func, **kwargs)

    # Otherwise, filter only the kwargs that match function's signature
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

    # TODO separate query from fitting?
    query_func: Callable = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=clf, fit_clf=True,
                                          discriminator=clf)
    return query_func


# region API
def get_dataset_config_options() -> dict[str, DatasetConfig]:
    config_options: dict[str, DatasetConfig] = parse_yaml_config_dir(DATA_CONFIG_PATH)
    return config_options


def get_qs_config_options() -> dict[str, QueryStrategyConfig]:
    return parse_yaml_config_dir(QS_CONFIG_PATH)


def get_model_config_options() -> dict[str, ModelConfig]:
    return parse_yaml_config_dir(MODEL_CONFIG_PATH)


def request_query(
    cfg: ActiveMlConfig,
    session_cfg: SessionConfig,
    X: np.ndarray,
    file_names: list[str],
) -> np.ndarray:

    labels = _load_or_init_annotations(X, file_names, cfg.dataset.id)

    query_func = _setup_query(cfg, session_cfg)
    print("Querying the active ML model ...")
    # If the query function expects an array, create one based on ordered indices.
    query_indices = query_func(X=X, y=labels)
    return query_indices


def _deserialize_annotations(json_path: Path) -> list[Annotation]:
    with json_path.open('r') as f:
        annotations_data: dict = json.load(f)

    annotations = [Annotation(**ann) for ann in annotations_data]
    return annotations


def _serialize_annotations(path: Path, annotations: list[Annotation]):
    with path.open("w") as f:
        json.dump([asdict(ann) for ann in annotations], f, indent=4)


def _load_labels(json_path: Path) -> np.ndarray:
    """Load labels from a JSON file and return as a numpy array."""
    with json_path.open('r') as f:
        annotations = json.load(f)
        num_annotations = len(annotations)

        labels = np.empty(num_annotations, dtype=float)

        # TODO check if there is still some missing labels.
        # Else there is nothing more to label.
        for i, ann in enumerate(annotations):
            labels[i] = ann['label']

    return labels


def _load_or_init_annotations(
    X: np.ndarray,
    file_names: list[str],
    dataset_id: str
) -> np.ndarray:
    """Load existing labels or initialize with missing labels."""
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'

    if json_file_path.exists():
        labels = _load_labels(json_file_path)
    else:
        # No annotations yet, initialize new JSON File putting labels to missing.
        annotations = [Annotation(file_name=f_name, label=MISSING_LABEL) for f_name in file_names]
        _serialize_annotations(json_file_path, annotations)
        # Initialize all labels to missing.
        labels = np.full(len(X), MISSING_LABEL, dtype=float)

    return labels


def completed_batch(dataset_id: str, batch: Batch):
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'

    print("completed batch")
    print(json_file_path)

    if not json_file_path.exists():
        raise RuntimeError("JSON file should already exist here!")

    annotations: list[Annotation] = _deserialize_annotations(json_file_path)

    # Update labeled data with new annotations.
    # Use string keys to avoid duplicates.
    for idx, annotation_val in zip(batch.indices, batch.annotations):
        # labels[idx] = annotation_val
        annotations[idx].label = annotation_val

    # Write back the updated labels overwriting existing file.
    _serialize_annotations(json_file_path, annotations)
# endregion


def finish_label_session():
    # Notify UI that. There is not more labels to label.
    raise NotImplementedError


def stop_labelling_session():
    raise NotImplementedError


def retrain():
    raise NotImplementedError
