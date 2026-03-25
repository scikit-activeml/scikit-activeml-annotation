import re
from io import BytesIO
import base64
from typing import TypeVar, TypeGuard, cast
from collections import OrderedDict
from collections.abc import Sequence
from itertools import islice
import json
import inspect
from functools import partial, lru_cache
from pathlib import Path
from typing import Callable
import logging

import hydra
import pydantic
import omegaconf
from omegaconf import OmegaConf

import numpy as np
import numpy.typing as npt

import sklearn
from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from sklearn.preprocessing import LabelEncoder

from skactiveml_annotation import util
from skactiveml_annotation.util import deserialize
import skactiveml_annotation.paths as sap

from skactiveml_annotation.core.schema import (
    SessionConfig,
    Annotation,
    AutomatedAnnotation,
    Batch,
    MISSING_LABEL_MARKER,
    DISCARD_MARKER,
    HistoryIdx,
)

from skactiveml_annotation.hydra_schema import (
    ActiveMlConfig,
    EmbeddingConfig,
    QueryStrategyConfig,
    ModelConfig,
    DatasetConfig,
)

from skactiveml_annotation.core.shared_types import DashProgressFunc
from skactiveml_annotation.util.utils import SortOrder

QueryFunc = Callable[..., npt.NDArray[np.intp]]

T = TypeVar("T")

def not_none_type_narrowing(x: T | None) -> TypeGuard[T]:
    return x is not None

def get_dataset_config_options() -> list[DatasetConfig]:
    return deserialize.parse_yaml_config_dir(sap.DATA_CONFIG_PATH, DatasetConfig)

def get_qs_config_options() -> list[QueryStrategyConfig]:
    return deserialize.parse_yaml_config_dir(sap.QS_CONFIG_PATH, QueryStrategyConfig)

def get_model_config_options() -> list[ModelConfig]:
    return deserialize.parse_yaml_config_dir(sap.MODEL_CONFIG_PATH, ModelConfig)

def get_embedding_config_options() -> list[EmbeddingConfig]:
    return deserialize.parse_yaml_config_dir(sap.EMBEDDING_CONFIG_PATH, EmbeddingConfig)

def get_query_cfg_from_id(query_id: str) -> QueryStrategyConfig:
    path = sap.QS_CONFIG_PATH / f'{query_id}.yaml'
    return deserialize.parse_yaml_file(path, QueryStrategyConfig)

def get_dataset_cfg_from_id(id: str) -> DatasetConfig:
    path = sap.DATA_CONFIG_PATH / f'{id}.yaml'
    return deserialize.parse_yaml_file(path, DatasetConfig)

def get_dataset_cfg_from_path(path: Path) -> DatasetConfig:
    return deserialize.parse_yaml_file(path, DatasetConfig)

def get_dataset_omegaconf_from_id(dataset_id: str) -> omegaconf.DictConfig | omegaconf.ListConfig:
    path = sap.DATA_CONFIG_PATH / f"{dataset_id}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")

    return OmegaConf.load(path)


def is_dataset_embedded(dataset_id: str, embedding_id: str) -> bool:
    key = f"{dataset_id}_{embedding_id}"
    path = sap.EMBEDDINGS_CACHE_PATH / f"{key}.npz"
    return path.exists()

def dataset_path_exits(dataset_path: str) -> bool:
    path = sap.ROOT_PATH / dataset_path
    return path.exists()


@lru_cache(maxsize=1)
def compose_config(overrides: tuple[tuple[str, str], ...]) -> ActiveMlConfig:
    overrides_hydra = deserialize.overrides_to_list(overrides)

    with hydra.initialize_config_dir(version_base=None, config_dir=str(sap.CONFIG_PATH)):
        cfg = hydra.compose('config', overrides=overrides_hydra)

        overrides_dict = dict(overrides)
        dataset_id = overrides_dict.get('dataset')
        if dataset_id is None:
            raise KeyError(
                "Missing key 'dataset' in overrides. Dataset selection must be provided"
            )

        # Check if dataset was overriden if for instance additional labels
        # have been added swap out dataset config to access that data
        if deserialize.is_dataset_cfg_overridden(dataset_id):
            path = sap.OVERRIDE_CONFIG_DATASET_PATH / f'{dataset_id}.yaml'
            cfg.dataset = OmegaConf.load(path)

        deserialize.set_ids_from_overrides(cfg, overrides)

        try:
            return ActiveMlConfig.model_validate(cfg)
        except pydantic.ValidationError as e:
            logging.error(f"Could not parse hydra configuration as ActiveMlConfig with error: {e}")
            raise


def _get_sklearn_classes(clf: SkactivemlClassifier) -> list[str]:
    """
    Return classifier classes as a list of strings.

    Raises:
        ValueError: If the classifier is not fitted or has no valid classes_.
    """
    try:
        raw_classes = clf.classes_
    except AttributeError as e:
        raise ValueError(
            f"{type(clf).__name__} has no 'classes_' attribute. "
            "Is the model fitted?"
        ) from e

    return np.asarray(raw_classes, dtype=str).tolist()


def request_query(
    cfg: ActiveMlConfig,
    session_cfg: SessionConfig,
    X: np.ndarray,
    filter_out_emb_indices: list[int] | None = None,
) -> Batch:
    y = _load_or_init_annotations(X, cfg.dataset)

    if filter_out_emb_indices is not None:
        # Exclude these embedding indices from querying by marking them as discarded
        y[filter_out_emb_indices] = DISCARD_MARKER

    query_func, clf = _setup_query(cfg, session_cfg)

    # Only fit and query on the samples not marked as discarded
    X_cand, y_cand, cand_to_emb_idx = _filter_discarded_samples(X, y)

    logging.info("Fitting the classifier")

    # The model is wrapped. Rely on scikit-activeml error handling
    # Any errors during fitting will result in fallback predictions using class label counts.
    clf.fit(X_cand, y_cand)

    logging.info("Querying the active ML model ...")

    try:
        query_cand_indices = query_func(X=X_cand, y=y_cand)
    except Exception as e:
        raise RuntimeError(
            f'Sample selection process failed with error: {e}'
        )

    # Map back cand indices to embedding index space
    emb_indices = cand_to_emb_idx[query_cand_indices]
    query_embeddings = X[emb_indices]

    try:
        class_probas = _safe_predict_proba(clf, query_embeddings)
    except Exception as e:
        logging.warning(
            f"No class probabilities can be displayed."
            f"predict_proba failed with error error: {e}. "
        )
        class_probas = None

    classes_sklearn = _get_sklearn_classes(clf)

    emb_indices = emb_indices.astype(int).tolist()

    # Possibly restore annotations that have been previously skipped
    file_paths = get_file_paths(cfg.dataset.id, cfg.embedding.id, emb_indices)
    annotations_data = _deserialize_annotations(cfg.dataset.id)

    # file_paths are the keys
    annotations = [
        annotations_data.get(f_path) for f_path in file_paths
    ]

    return Batch(
        emb_indices=emb_indices,
        class_probas=class_probas,
        classes_sklearn=classes_sklearn,
        progress=0,
        annotations=annotations,
    ).init()


def _safe_predict_proba(
    clf: SkactivemlClassifier,
    emb_samples: np.ndarray,
) -> list[list[float]]:
    """
    Call predict_proba on a classifier.

    Raises:
        AttributeError: if predict_proba is not supported.
        RuntimeError: if predict_proba fails at runtime.
    """
    if not hasattr(clf, "predict_proba"):
        raise AttributeError(
            f"{clf.__class__.__name__} does not support predict_proba"
        )

    try:
        class_probas = clf.predict_proba(emb_samples)
    except Exception as e:
        raise RuntimeError(
            f"predict_proba failed for {clf.__class__.__name__}"
        ) from e

    # scikit-learn guarantees predict_proba returns an array-like of shape
    # (n_samples, n_classes). Calling tolist() therefore produces
    # list[list[float]].
    return class_probas.tolist()


def compute_embeddings(
    activeml_cfg: ActiveMlConfig,
    progress_func: DashProgressFunc
):
    embedding_cfg = activeml_cfg.embedding
    dataset_cfg = activeml_cfg.dataset
    dataset_id = dataset_cfg.id

    data_path = Path(dataset_cfg.data_path)
    if not data_path.is_absolute():
        data_path = sap.ROOT_PATH / data_path

    adapter = embedding_cfg.definition.instantiate()
 
    X, file_paths = adapter.compute_embeddings(data_path, progress_func)

    file_paths_str = _normalize_and_validate_paths(file_paths, X)

    # Unique key
    cache_key = f"{dataset_id}_{embedding_cfg.id}"
    cache_path = sap.EMBEDDINGS_CACHE_PATH / f"{cache_key}.npz"  # Use .npz to store multiple arrays

    logging.info(f"Embedding has been computed and saved at {cache_path}")

    # Store relative file_paths
    np.savez(cache_path, X=X, file_paths=file_paths_str)


@lru_cache(maxsize=1)
def load_embeddings(
    dataset_id: str,
    embedding_id: str,
) -> np.ndarray:
    cache_key = f"{dataset_id}_{embedding_id}"
    cache_path = sap.EMBEDDINGS_CACHE_PATH / f"{cache_key}.npz"

    if not cache_path.exists():
        raise RuntimeError(f"Cannot get embedding at path: {cache_path}! \nEmbedding should exists already")

    with np.load(str(cache_path)) as data:
        X = data['X']
    return X


def get_total_num_samples(dataset_id: str, embedding_id: str) -> int:
    return len(load_embeddings(dataset_id, embedding_id))


def auto_annotate(
    X: np.ndarray,
    cfg: ActiveMlConfig,
    threshold: float,
    sort_by_proba: bool = True
):
    y = _load_or_init_annotations(X, cfg.dataset)

    model_cfg = cfg.model
    if model_cfg is None:
        logging.warning("Cannot auto complete as there is no classifier selected!")
        return

    random_state = np.random.RandomState(cfg.random_seed)
    clf = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    # Fit classifier on samples not marked as discarded
    X_cand, y_cand, _ = _filter_discarded_samples(X, y)
    clf.fit(X_cand, y_cand)

    # Auto Annotate all samples that were not annotated and where the 
    # top class probability meets the threshold
    X_missing, _, mapping = _filter_out_annotated(X, y)
    class_probas = clf.predict_proba(X=X_missing)  # shape (num_samples * num_labels)

    top_indices = np.argmax(class_probas, axis=1)

    assert clf.classes_ is not None
    top_classes = clf.classes_[top_indices]
    # Select top proba from each row
    top_probas = class_probas[np.arange(class_probas.shape[0]), top_indices]

    # Select samples that are above the threshold probability
    is_threshold = (top_probas > threshold)
    emb_indices = mapping[is_threshold]
    classes = top_classes[is_threshold]
    probas = top_probas[is_threshold]

    if sort_by_proba:
        # Negate for descending order
        sorted_indices = np.argsort(-probas)
        emb_indices = emb_indices[sorted_indices]
        classes = classes[sorted_indices]
        probas = probas[sorted_indices]

    # Get file paths for embedding indices
    selected_file_paths = get_file_paths(
        cfg.dataset.id,
        cfg.embedding.id,
        emb_indices=emb_indices,
    )

    # python lists garantee to preserve insertion order since pytyon 3.17
    auto_annots = {
        f_path: AutomatedAnnotation(
            embedding_idx=int(emb_idx),
            label=label,
            confidence=float(proba),
        )
        for f_path, (emb_idx, label, proba)
        in zip(selected_file_paths, zip(emb_indices, classes, probas))
    }

    manual_annots = _deserialize_annotations(cfg.dataset.id)

    json_store_path = sap.ANNOTATED_PATH / f'{cfg.dataset.id}-automated.json'
    _serialize_automatic_and_manual_annotations(
        json_store_path,
        manual_annots,
        auto_annots,
    )

    num_auto_annotated = len(auto_annots)
    num_total_annotated = num_auto_annotated + len(manual_annots)
    logging.info(f'{num_auto_annotated} samples have been automatically annoted @\n{json_store_path}')
    logging.info(f'In total annotated: {num_total_annotated}')


def add_class(
    dataset_cfg: DatasetConfig,
    new_class_name: str,
    batch: Batch,
):
    # Validate new class name
    if new_class_name == '':
        raise ValueError("Class name has to have at least length 1")

    classes = dataset_cfg.classes

    if new_class_name in classes:
        raise ValueError(f"Cannot add new class because '{new_class_name}' already exists.")

    _add_class_and_save_yaml_override(classes, new_class_name, dataset_cfg.id)
    _update_batch_after_class_insertion(batch, classes, new_class_name)

    # Invalidate cache. Force new composing when called next time.
    compose_config.cache_clear()


def _add_class_and_save_yaml_override(
    classes: list[str],
    new_class_name: str,
    dataset_id: str,
):
    """
    Add a class to the dataset YAML config. Infer whether the existing classes
    are sorted and maintain that order if possible; otherwise, append the new
    class to the end. Save the updated dataset YAML config to the dataset
    override directory.

    Args:
        classes: Existing dataset class names.
        new_class_name: Class name to add.
        dataset_id: Dataset identifier used to load and save the config.
    """

    # Infer if the classes in the yaml file have an order
    sort_key = (
        float
        if util.utils.is_all_numeric(classes + [new_class_name])
        else None
    )
    sort_order = util.utils.get_sort_order(classes, sort_key)

    updated_classes = classes.copy()
    updated_classes.append(new_class_name)

    match sort_order:
        case SortOrder.ASC:
            updated_classes.sort(key=sort_key)
        case SortOrder.DESC:
            updated_classes.sort(key=sort_key, reverse=True)
        case SortOrder.UNSORTED:
            pass

    omega_cfg = get_dataset_omegaconf_from_id(dataset_id)
    omega_cfg.classes = updated_classes

    # Derive override path
    sap.OVERRIDE_CONFIG_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    override_path = (
        sap.OVERRIDE_CONFIG_DATASET_PATH / f"{dataset_id}.yaml"
    )

    OmegaConf.save(config=omega_cfg, f=override_path)


def _update_batch_after_class_insertion(
    batch: Batch,
    classes: list[str],
    new_class_name: str,
):
    """
    Update a Batch after inserting a new class.

    This inserts the new class into the sklearn class list at the correct
    position and updates the class probability matrix if present.
    """

    sklearn_insertion_idx = _insertion_index_sklearn(classes, new_class_name)
    batch.classes_sklearn.insert(sklearn_insertion_idx, new_class_name)

    if batch.class_probas is not None:
        batch.class_probas = _insert_class_prob_column(
            batch.class_probas,
            sklearn_insertion_idx,
        )


def _insertion_index_sklearn(
    classes: list[str],
    new_class: str,
) -> int:
    label_enc = LabelEncoder()
    label_enc.fit(classes + [new_class])
    insertion_idx = label_enc.transform([new_class])[0]
    return insertion_idx


def _insert_class_prob_column(probas: list[list[float]], idx: int) -> list[list[float]]:
    return [
        row[:idx] + [0.0] + row[idx:]
        for row in probas
    ]


def _load_or_init_annotations(
    X: np.ndarray,
    dataset_cfg: DatasetConfig,
) -> np.ndarray:
    """Load existing labels or initialize with missing labels."""
    num_samples = len(X)
    max_label_name_len = max(
        len(s)
        for s in dataset_cfg.classes + [DISCARD_MARKER, MISSING_LABEL_MARKER]
    )

    y = np.full(num_samples, MISSING_LABEL_MARKER, dtype=f'U{max_label_name_len}')

    _load_labels_as_np(y, dataset_cfg.id)

    return y


def get_num_annotated(dataset_id: str, exclude_missing: bool = False) -> int:
    annotations = _deserialize_annotations(dataset_id)

    if not exclude_missing:
        return len(annotations)

    return sum(
        (
            1
            for annot in annotations.values()
            if annot.label != MISSING_LABEL_MARKER
        )
    )


def update_json_annotations(
    dataset_id: str,
    embedding_id: str,
    batch: Batch,
):
    file_paths = get_file_paths(dataset_id, embedding_id, batch.emb_indices)

    update_annotations(
        dataset_id,
        file_paths,
        batch.annotations,
    )

    # Assumes the idx is on the first of the current batch
    # Put the idx on the last element of the batch
    increment_global_history_idx(dataset_id, len(batch.annotations) - 1)


def _deserialize_annotations(dataset_id: str) -> OrderedDict[str, Annotation]:
    json_path = sap.ANNOTATED_PATH / f"{dataset_id}.json"

    # If the file doesn't exist or is empty → return an empty OrderedDict
    if not json_path.exists() or json_path.stat().st_size == 0:
        return OrderedDict()

    content = json_path.read_text().strip()
    if not content:
        return OrderedDict()

    annotations_data: dict = json.loads(content)

    return OrderedDict(
        (key, Annotation.model_validate(ann_data))
        for key, ann_data in annotations_data.items()
    )

def _serialize_annotations(dataset_id: str, annotations: OrderedDict[str, Annotation]):
    json_path = sap.ANNOTATED_PATH / f"{dataset_id}.json"

    with json_path.open("w") as f:
        json.dump(
            OrderedDict(
                [(key, ann.model_dump(mode='json')) for key, ann in annotations.items()]
            ),
            f,
            indent=4
        )

# TODO: is this method needed when there is update_annotation_json()
def update_annotations(
    dataset_id: str,
    file_paths: list[str],
    new_annotations: Sequence[Annotation | None],
):
    annotations = _deserialize_annotations(dataset_id)

    # Get file_paths as they are the keys
    new_annotations_dict = OrderedDict(
        (f_path, annot) for f_path, annot in zip(file_paths, new_annotations)
        if annot is not None
    )

    annotations.update(new_annotations_dict)
    _serialize_annotations(dataset_id, annotations)


def _serialize_automatic_and_manual_annotations(
    path: Path,
    manual_annotations: dict[str, Annotation],
    auto_annotations: dict[str, AutomatedAnnotation],
):
    payload = {
        'manual': {
            f_path: ann.model_dump(mode='json') for f_path, ann in manual_annotations.items()
        },
        'automatic': {
            f_path: ann.model_dump(mode='json') for f_path, ann in auto_annotations.items()
        },
    }

    path.write_text(
        json.dumps(payload, indent=4),
        encoding='utf-8',
    )


def _load_labels_as_np(y: np.ndarray, dataset_id: str):
    """Load labels from a JSON file and return as a numpy array."""
    annotations = _deserialize_annotations(dataset_id)

    num_annotations = len(annotations)
    emb_indices = np.empty(num_annotations, dtype=int)
    labels = np.empty(num_annotations, dtype=object)

    for i, ann in enumerate(annotations.values()):
        emb_indices[i] = ann.embedding_idx
        labels[i] = ann.label
    
    y[emb_indices] = labels


def _clf_accepts_random(clf_cls) -> bool:
    sig = inspect.signature(clf_cls.__init__)
    return "random_state" in sig.parameters


def _filter_discarded_samples(X: npt.NDArray[np.number], y: npt.NDArray[np.number]):
    # keep = np.isfinite(y) | np.isnan(y)  # np.isfinite(np.nan) == False
    keep = (y != DISCARD_MARKER)
    X_filtered = X[keep]
    y_filtered = y[keep]
    mapping = np.arange(len(X))[keep]
    return X_filtered, y_filtered, mapping


def _filter_out_annotated(X: npt.NDArray[np.number], y: npt.NDArray[np.number]):
    missing = (y == MISSING_LABEL_MARKER)
    X_filtered = X[missing]
    y_filtered = y[missing]
    mapping = np.arange(len(X))[missing]
    return X_filtered, y_filtered, mapping


def _build_activeml_classifier(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    random_state: np.random.RandomState
) -> SkactivemlClassifier:
    classes = dataset_cfg.classes
    # n_classes = len(dataset_cfg.classes)
    # classes = np.arange(n_classes)

    clf_cls = model_cfg.definition.target_

    kwargs = {}
    if _clf_accepts_random(clf_cls):
        kwargs['random_state'] = random_state

    clf = model_cfg.definition.instantiate(**kwargs)

    if isinstance(clf, SkactivemlClassifier):
        # Classifier is already wrapped aka supports missing labels
        # assigning a string for missing_label is valid from the documentation.
        clf.missing_label=MISSING_LABEL_MARKER  # pyright: ignore[reportAttributeAccessIssue]
        return clf
    elif isinstance(clf, sklearn.base.ClassifierMixin):
        wrapped_clf = SklearnClassifier(
            estimator=clf,
            classes=classes,
            random_state=random_state,
            missing_label=MISSING_LABEL_MARKER,  # pyright: ignore[reportArgumentType]
        )
        return wrapped_clf
    else:
        raise RuntimeError(f"Classifer is not a sklearn ClassifierMixin")


def _filter_kwargs(func: QueryFunc, **kwargs) -> QueryFunc:
    params = inspect.signature(func).parameters
    param_names = params.keys()

    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_kwargs:
        # If the func accepts **kwargs, no filtering is needed
        return partial(func, **kwargs)

    # Otherwise, filter only the kwargs that match function's signature
    filtered_kwargs = {p_name: p_obj for p_name, p_obj in kwargs.items() if p_name in param_names}

    return partial(func, **filtered_kwargs)


def _setup_query(cfg: ActiveMlConfig, session_cfg: SessionConfig) -> tuple[QueryFunc, SkactivemlClassifier]:
    random_state = np.random.RandomState(cfg.random_seed)

    model_cfg = cfg.model
    clf = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    # max_candidates for subsampling.
    qs = cfg.query_strategy.definition.instantiate(
        random_state = random_state,
        missing_label = MISSING_LABEL_MARKER
    )

    if session_cfg.subsampling is not None:
        qs = SubSamplingWrapper(
            qs,
            max_candidates=session_cfg.subsampling,
            random_state=random_state,
            # from doc missing_label: scalar | str | np.nan | None,
            missing_label=MISSING_LABEL_MARKER,  # pyright: ignore[reportArgumentType]
        )

    # Dont fit classifier here to prevent fitting twice
    query_func = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=clf, fit_clf=False,
                                          discriminator=clf)
    return query_func, clf


def _normalize_and_validate_paths(
    file_paths: list[Path],
    X: np.ndarray,
) -> list[str]:
    if len(file_paths) != len(X):
        raise RuntimeError(f'Amount of samples does not match amount of file paths!')

    file_paths_str: list[str] = []
    has_absolute = False

    for p in file_paths:
        if p.is_absolute():
            has_absolute = True
            if not p.is_file():
                raise RuntimeError(f'path does not exist or is not a file: {p}')

            p_str = str(p)
        else:
            full_p = sap.ROOT_PATH / p
            if not full_p.is_file():
                raise RuntimeError(
                    f'Resolved path: {full_p} does not exist or is not a file.\n'
                    f'resolved from relative path: {p}'
                )

            p_str = p.as_posix()

        file_paths_str.append(p_str)

    if has_absolute:
        logging.warning("absolute paths were provided. Results won't be easily shareable")
    return file_paths_str


def get_one_file_path(
    dataset_id: str,
    embedding_id: str,
    emb_idx: int
) -> str:
    return get_file_paths(dataset_id, embedding_id, emb_idx)[0]


def get_file_paths(
    dataset_id: str,
    embedding_id: str,
    emb_indices: np.ndarray[tuple[int], np.dtype[np.intp]] | list[int] | int,
) -> list[str]:
    cache_key = f'{dataset_id}_{embedding_id}'
    cache_path = sap.EMBEDDINGS_CACHE_PATH / f"{cache_key}.npz"

    if isinstance(emb_indices, int):
        emb_indices = [emb_indices]

    if not cache_path.exists():
        raise RuntimeError(f"Cannot get embedding at path: {cache_path}! \nEmbedding should exists already")

    with np.load(str(cache_path), mmap_mode='r') as data:
        file_paths = data['file_paths']
        # tolist() returns np.array if given a list
        return file_paths[emb_indices].tolist()


def ensure_global_history_idx_init(dataset_id: str):
    try:
        global_history_idx = get_global_history_idx(dataset_id)
        return
    except FileNotFoundError:
        history_size = get_num_annotated(dataset_id)
        if history_size == 0:
            global_history_idx = 0
        else:
            # Assume there have been annotations made but the index is missing
            global_history_idx = history_size - 1
            
        set_global_history_idx(dataset_id, global_history_idx)


def get_global_history_idx(dataset_id: str) -> int:
    """
    Retrieve the history index for a given dataset ID.
    Returns None if the file dose not exist
    """
    path = sap.HISTORY_IDX / f"{dataset_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Global history index not found at {path}. This file should already exist."
        )

    # Read JSON from file
    content = path.read_text()
    model = HistoryIdx.model_validate_json(content)
    return model.idx


def set_global_history_idx(dataset_id: str, value: int) -> None:
    """
    Store (or update) the history index for a given dataset ID.
    Creates the directory if needed.
    """
    path = sap.HISTORY_IDX / f"{dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    model = HistoryIdx(idx=value)

    path.write_text(model.model_dump_json(indent=4))


def increment_global_history_idx(dataset_id: str, value: int):
    current_idx = get_global_history_idx(dataset_id)
    new_val = current_idx + value
    set_global_history_idx(dataset_id, new_val)
    

def restore_batch(
    cfg: ActiveMlConfig,
    history_idx: int,
    restore_forward: bool,
    num_restore: int,
) -> Batch:
    # When restoring backwards it will try to restore num_restore samples
    # If there are not enough samples left to restore it will restore as much as it can
    # If it cant restore it will throw an error
    # Assumes annotations are stored in json in the same order they were made.
    logging.info("\nRestore Batch")
    # History_idx is exclusive and wont be restored

    if restore_forward:
        start = history_idx + 1
        end = start + num_restore
    else:
        end = history_idx # exclusive
        start = max(0, end - num_restore)

        num_restorable = end - start

        if num_restorable <= 0:
            logging.info(f"There is no samples left to restore backwards")
            raise RuntimeError()

        elif num_restorable < num_restore:
            logging.info(f"Can not restore backwards {num_restore}, only {num_restorable} will be restored")
        
    annotations_data = _deserialize_annotations(cfg.dataset.id)
    sliced = islice(annotations_data.values(), start, end)
    annotations = list(sliced)

    emb_indices = [annot.embedding_idx for annot in annotations]
    
    model_cfg = cfg.model
    random_state = np.random.RandomState(cfg.random_seed)
    clf = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    X = load_embeddings(cfg.dataset.id, cfg.embedding.id)
    y = _load_or_init_annotations(X, cfg.dataset)
    X_cand, y_cand, _ = _filter_discarded_samples(X, y)

    clf.fit(X_cand, y_cand)
    class_probas = clf.predict_proba(X[emb_indices])

    return Batch(
        emb_indices=emb_indices,
        class_probas=class_probas.tolist(),
        classes_sklearn=_get_sklearn_classes(clf),
        progress=0 if restore_forward else len(emb_indices) - 1,
        annotations=cast(list[Annotation | None], annotations),
    ).init()


def file_buffer_to_inline_data_url(file_data_buffer: BytesIO, mime: str) -> str:
    b64_encoded_file_data = base64.b64encode(file_data_buffer.getvalue()).decode()
    return f"data:{mime};base64,{b64_encoded_file_data}"


def camel_case_to_title(s: str) -> str:
    # Split before capital letters that are followed by lowercase (normal word start)
    # or when a lowercase is followed by a capital (e.g., "HTMLParser")
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', s)
    return " ".join(parts)
