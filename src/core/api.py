import inspect
from pathlib import Path
from typing import cast
from inspect import signature
from functools import partial, lru_cache
from typing import Callable

import numpy as np
from hydra.utils import instantiate

from skactiveml.utils import MISSING_LABEL
from skactiveml.pool import SubSamplingWrapper
from skactiveml.classifier import SklearnClassifier
from skactiveml.base import (
    QueryStrategy,
    SkactivemlClassifier
)

from sklearn.base import ClassifierMixin

from core.schema import *
from embedding.base import EmbeddingBaseAdapter

from util.deserialize import (
    parse_yaml_config_dir,
    parse_yaml_file
)
from paths import (
    DATA_CONFIG_PATH,
    ANNOTATED_PATH,
    QS_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    EMBEDDING_CONFIG_PATH,
    DATASETS_PATH,
    CACHE_PATH,
    ROOT_PATH
)


# region API
def get_dataset_config_options() -> list[DatasetConfig]:
    cfgs = parse_yaml_config_dir(DATA_CONFIG_PATH)
    return cast(list[DatasetConfig], cfgs)


def get_qs_config_options() -> list[QueryStrategyConfig]:
    cfgs = parse_yaml_config_dir(QS_CONFIG_PATH)
    return cast(list[QueryStrategyConfig], cfgs)


def get_model_config_options() -> list[ModelConfig]:
    cfgs = parse_yaml_config_dir(MODEL_CONFIG_PATH)
    return cast(list[ModelConfig], cfgs)


def get_embedding_config_options() -> list[EmbeddingConfig]:
    cfgs = parse_yaml_config_dir(EMBEDDING_CONFIG_PATH)
    return cast(list[EmbeddingConfig], cfgs)


def get_query_cfg_from_id(query_id) -> QueryStrategyConfig:
    cfg = parse_yaml_file(QS_CONFIG_PATH / f'{query_id}.yaml')
    return cast(QueryStrategyConfig, cfg)


def is_dataset_embedded(dataset_id, embedding_id) -> bool:
    key = f"{dataset_id}_{embedding_id}"
    path = Path(str(CACHE_PATH)) / f"{key}.npz"
    return path.exists()


def dataset_path_exits(dataset_path: str) -> bool:
    path = ROOT_PATH / Path(dataset_path)
    return path.exists()


def request_query(
        cfg: ActiveMlConfig,
        session_cfg: SessionConfig,
        X: np.ndarray,
) -> Batch:
    y = _load_or_init_annotations(X, cfg.dataset.id)
    query_func, clf = _setup_query(cfg, session_cfg)

    # Only fit and query on the samples not marked as outliers
    X_cand, y_cand, mapping = _filter_outliers(X, y)

    # TODO fitting classifier here might negate Subsampling QS wrapper
    if clf is not None:
        print("Fitting the classifier")
        # TODO can fitting the classifier fail?
        clf.fit(X_cand, y_cand)

    print("Querying the active ML model ...")

    try:
        query_indices_cand = query_func(X=X_cand, y=y_cand)
    except Exception as e:
        # TODO add error handling, UI notification and logging.
        raise RuntimeError(
            f'[ERROR] Sample selection process failed with error: {e}'
        )

    # Map back to original indices.
    # TODo naming sometimes query_indices sometimes embedding. Inconsistent.
    query_indices = mapping[query_indices_cand]

    # TODO sometimes query returns list of np.int64? It has be be serializeable in current implementation.
    if isinstance(query_indices, np.ndarray):
        query_indices = query_indices.tolist()
    if not isinstance(query_indices[0], int):
        query_indices = [int(x) for x in query_indices]

    query_samples = X[query_indices]

    if clf is None:
        class_probas = np.empty(0)
    else:
        # TODO clf could not have predict_proba
        class_probas = clf.predict_proba(query_samples)

    num_labeled_samples = np.sum(~np.isnan(y))
    total = X.shape[0]

    batch_state = Batch(
        indices=query_indices,
        class_probas=class_probas.tolist(),
        progress=0,
        annotations=[None] * len(query_indices)
    )
    print("\nNew Batch decided")
    print(f"Labaled/total: {num_labeled_samples} / {total}")
    return batch_state


def compute_embeddings(
        activeml_cfg: ActiveMlConfig,
        progress_func: callable
):
    embedding_cfg = activeml_cfg.embedding
    dataset_cfg = activeml_cfg.dataset
    dataset_id = dataset_cfg.id

    data_path = dataset_cfg.data_path
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = ROOT_PATH / data_path

    adapter: EmbeddingBaseAdapter = instantiate(embedding_cfg.definition)

    X, file_paths = adapter.compute_embeddings(data_path, progress_func)

    file_paths_str = _normalize_and_validate_paths(file_paths, X)

    # Unique key
    cache_key = f"{dataset_id}_{embedding_cfg.id}"
    cache_path = CACHE_PATH / f"{cache_key}.npz"  # Use .npz to store multiple arrays

    # Store relative file_paths
    np.savez(cache_path, X=X, file_paths=file_paths_str)


@lru_cache(maxsize=1)
def load_embeddings(
        dataset_id: str,
        embedding_id: str,
) -> np.ndarray:

    cache_key = f"{dataset_id}_{embedding_id}"
    cache_path = CACHE_PATH / f"{cache_key}.npz"

    if not cache_path.exists():
        raise RuntimeError(f"Cannot get embedding at path: {cache_path}! \nEmbedding should exists already")

    with np.load(str(cache_path)) as data:
        X = data['X']
    return X


# TODO rename to update json_annotations
def completed_batch(dataset_id: str, batch: Batch, embedding_id: str) -> int:
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'
    print("completed batch")
    print(json_file_path)

    # Get existing annotations
    annotations: list[Annotation] = _deserialize_annotations(json_file_path)
    file_paths = load_file_paths(dataset_id, embedding_id)
    print(batch.annotations)

    # Create new Annotations
    new_annotations = [
        Annotation(
            embedding_idx=idx,  # TODO use better names
            file_name=file_paths[idx],  # TODO store unix like file paths.
            label=annot_val
        )
        for idx, annot_val in zip(batch.indices, batch.annotations)
        if not np.isnan(annot_val)  # Do not store missing LABEL
    ]

    updated_annotations = annotations + new_annotations

    # Override annotations
    _serialize_annotations(json_file_path,  updated_annotations)

    num_annotated = len(updated_annotations)
    return num_annotated


def get_num_annotated(dataset_id: str) -> int:
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'
    return len(_deserialize_annotations(json_file_path))


def get_total_num_samples(dataset_id, embedding_id) -> int:
    return len(load_embeddings(dataset_id, embedding_id))
# endregion


# TODO put this stuff into utils package?
def _load_or_init_annotations(
        X: np.ndarray,  # TODO Should not take X but rather num_of_samples
        dataset_id: str
) -> np.ndarray:
    """Load existing labels or initialize with missing labels."""
    json_file_path = ANNOTATED_PATH / f'{dataset_id}.json'
    num_samples = len(X)

    if json_file_path.exists():
        labels = _load_labels_as_np(num_samples, json_file_path)
    else:
        # Put all labels to missing.
        labels = np.full(num_samples, MISSING_LABEL, dtype=float)

    return labels


def _deserialize_annotations(json_path: Path) -> list[Annotation]:
    if not json_path.exists():
        return []

    with json_path.open('r') as f:
        annotations_data: dict = json.load(f)

    annotations = [Annotation(**ann) for ann in annotations_data]
    return annotations


def _serialize_annotations(path: Path, annotations: list[Annotation]):
    with path.open("w") as f:
        json.dump([asdict(ann) for ann in annotations], f, indent=4)


def _load_labels_as_np(num_samples: int, json_path: Path) -> np.ndarray:
    """Load labels from a JSON file and return as a numpy array."""
    with json_path.open('r') as f:
        annotations = json.load(f)

        labels = np.full(num_samples, MISSING_LABEL, dtype=float)

        # TODO check if there is still some missing labels.
        # Else there is nothing more to label.
        for ann in annotations:
            idx = ann['embedding_idx']
            labels[idx] = ann['label']

    return labels


def _estimator_accepts_random(est_cls) -> bool:
    sig = inspect.signature(est_cls.__init__)
    return "random_state" in sig.parameters


def _filter_outliers(X, y):
    mask = np.isfinite(y) | np.isnan(y)  # np.isfinite(np.nan) == False
    X_filtered = X[mask]
    y_filtered = y[mask]
    mapping = np.arange(len(X))[mask]
    return X_filtered, y_filtered, mapping


# TODO what api do the pages need from the api?
# TODO will this be used for estmiators aswell?
def _build_activeml_classifier(
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        random_state: np.random.RandomState
) -> SkactivemlClassifier:
    # classes = dataset_cfg.classes
    n_classes = len(dataset_cfg.classes)
    classes = np.arange(n_classes)

    # TODO rename to Estimator?
    est_cls = model_cfg.definition['_target_']

    kwargs = {}
    if _estimator_accepts_random(est_cls):
        kwargs['random_state'] = random_state

    est = instantiate(model_cfg.definition, **kwargs)

    if isinstance(est, SkactivemlClassifier):
        # Classifier is already wrapped aka supports missing labels
        return est
    elif isinstance(est, ClassifierMixin):
        wrapped_est = SklearnClassifier(
            estimator=est,
            classes=classes,
            random_state=random_state,
            # missing_label=schema.MISSING_LABEL_STR
        )
        return wrapped_est
    else:
        raise RuntimeError(f"Estimator is not a sklearn ClassifierMixin")


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
def _setup_query(cfg: ActiveMlConfig, session_cfg: SessionConfig) -> tuple[Callable, SklearnClassifier | None]:
    random_state = np.random.RandomState(cfg.random_seed)

    model_cfg = cfg.model
    if model_cfg is None:
        # TODO use estimator to have more accurate terminology
        estimator = None
    else:
        estimator: SklearnClassifier = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    # max_candidates for subsampling.
    qs: QueryStrategy = instantiate(cfg.query_strategy.definition, random_state=random_state)

    if session_cfg.subsampling is not None:
        qs: SubSamplingWrapper = SubSamplingWrapper(
            qs,
            max_candidates=session_cfg.subsampling,
            random_state=random_state
        )

    query_func: Callable = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=estimator, fit_clf=False,
                                          discriminator=estimator)
    return query_func, estimator


def _normalize_and_validate_paths(
        file_paths: list[str | Path],
        X: np.ndarray
) -> list[str]:
    if len(file_paths) != len(X):
        raise RuntimeError(f'Amount of samples does not match amount of file paths!')

    file_paths_str = []
    has_absolute = False

    for p in file_paths:
        if isinstance(p, str):
            p = Path(p)
        if p.is_absolute():
            has_absolute = True
            if not p.is_file():
                raise RuntimeError(f'path does not exist or is not a file: {p}')

            p_str = str(p)
        else:
            full_p = ROOT_PATH / p
            if not full_p.is_file():
                raise RuntimeError(
                    f'Resolved path: {full_p} does not exist or is not a file.\n'
                    f'resolved from relative path: {p}'
                )

            p_str = p.as_posix()

        file_paths_str.append(p_str)

    if has_absolute:
        print("[WARNING] absolute paths were provided. Results won't be easily shareable.")
    return file_paths_str


@lru_cache(maxsize=1)
def load_file_paths(
        dataset_id: str,
        embedding_id: str
) -> list[str]:
    cache_key = f'{dataset_id}_{embedding_id}'
    cache_path = CACHE_PATH / f"{cache_key}.npz"

    if not cache_path.exists():
        raise RuntimeError(f"Cannot get embedding at path: {cache_path}! \nEmbedding should exists already")

    with np.load(str(cache_path)) as data:
        file_paths = data['file_paths'].tolist()  # TODO is this tolist needed?
    return file_paths


def undo_annots_and_restore_batch(cfg: ActiveMlConfig, num_undo: int) -> Batch | None:
    # Assumes annotations are stored in json in the same order they were made.
    json_file_path = ANNOTATED_PATH / f'{cfg.dataset.id}.json'
    annotations = _deserialize_annotations(json_file_path)

    # Check there is enough annotations made to go back that far.
    num_annotations = len(annotations)
    if num_undo > num_annotations:
        if num_annotations == 0:
            return None

        num_undo = num_annotations

    write_back, reconstruct = annotations[:-num_undo], annotations[-num_undo:]

    json_file_path = ANNOTATED_PATH / f'{cfg.dataset.id}.json'
    _serialize_annotations(json_file_path, write_back)

    model_cfg = cfg.model
    if model_cfg is None:
        # TODO use estimator to have more accurate terminology
        estimator = None
    else:
        random_state = np.random.RandomState(cfg.random_seed)
        estimator = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    embedding_indices = [annot.embedding_idx for annot in reconstruct]

    if estimator is not None:
        X = load_embeddings(cfg.dataset.id, cfg.embedding.id)
        y = _load_or_init_annotations(X, cfg.dataset.id)
        X_cand, y_cand, _ = _filter_outliers(X, y)

        estimator.fit(X_cand, y_cand)
        class_probas = estimator.predict_proba(X[embedding_indices])
    else:
        class_probas = np.empty(0)

    # Restored Batch
    return Batch(
        indices=embedding_indices,
        annotations=[annot.label for annot in reconstruct],
        class_probas=class_probas.tolist(),
        progress=num_undo,
    )