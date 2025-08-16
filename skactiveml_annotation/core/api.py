from collections.abc import Sequence
import json
import logging
import inspect
import bisect
from dataclasses import asdict 
from functools import partial, lru_cache
from pathlib import Path
from typing import Callable, cast

import dash.exceptions

import hydra
from omegaconf import OmegaConf

import numpy as np
import numpy.typing as npt

from pydantic import ValidationError
import sklearn
from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper 

from skactiveml_annotation import util
import skactiveml_annotation.paths as sap

from skactiveml_annotation.embedding.base import EmbeddingBaseAdapter
from skactiveml_annotation.core.schema import (
    ActiveMlConfig,
    EmbeddingConfig,
    QueryStrategyConfig,
    ModelConfig, 
    EmbeddingTarget,
    DatasetConfig,
    SessionConfig,
    Annotation,
    AutomatedAnnotation,
    Batch,
    MISSING_LABEL_MARKER,
    DISCARD_MARKER,
)

# TODO: remove region
# region API
def get_dataset_config_options() -> list[DatasetConfig]:
    raw_cfgs = util.deserialize.parse_yaml_config_dir(sap.DATA_CONFIG_PATH)
    try:
        return [DatasetConfig.model_validate(cfg) for cfg in raw_cfgs]
    except RuntimeError:
        # TODO:: Error handling!
        raise

def get_qs_config_options() -> list[QueryStrategyConfig]:
    raw_cfgs = util.deserialize.parse_yaml_config_dir(sap.QS_CONFIG_PATH)
    try:
        return [QueryStrategyConfig.model_validate(cfg) for cfg in raw_cfgs]
    except RuntimeError:
        raise

def get_model_config_options() -> list[ModelConfig]:
    raw_cfgs = util.deserialize.parse_yaml_config_dir(sap.MODEL_CONFIG_PATH)

    try:
        return [ModelConfig.model_validate(cfg) for cfg in raw_cfgs]
    except RuntimeError:
        raise


def get_embedding_config_options() -> list[EmbeddingConfig]:
    raw_cfgs = util.deserialize.parse_yaml_config_dir(sap.EMBEDDING_CONFIG_PATH)
    try:
        return [EmbeddingConfig.model_validate(cfg) for cfg in raw_cfgs]
    except RuntimeError:
        raise

def get_query_cfg_from_id(query_id: str) -> QueryStrategyConfig:
    path = sap.QS_CONFIG_PATH / f'{query_id}.yaml'
    cfg_raw = util.deserialize.parse_yaml_file(path)

    # TODO: The work should be done inside the deserialize function
    try:
        return QueryStrategyConfig.model_validate(cfg_raw)
    except ValidationError:
        logging.error(f"Failed to parse config at {path} as {QueryStrategyConfig.__name__}")
        raise

def get_dataset_cfg_from_path(path: Path) -> DatasetConfig:
    cfg_raw = util.deserialize.parse_yaml_file(path)
    try:
        return DatasetConfig.model_validate(cfg_raw)
    except ValidationError:
        logging.error(f"Failed to parse config at {path} as {DatasetConfig.__name__}")
        raise


def is_dataset_embedded(dataset_id: str, embedding_id: str) -> bool:
    key = f"{dataset_id}_{embedding_id}"
    path = sap.EMBEDDINGS_CACHE_PATH / f"{key}.npz"
    return path.exists()


def dataset_path_exits(dataset_path: str) -> bool:
    path = sap.ROOT_PATH / dataset_path
    return path.exists()


@lru_cache(maxsize=1)
def compose_config(overrides: tuple[tuple[str, str], ...]) -> ActiveMlConfig:
    overrides_hydra = util.deserialize.overrides_to_list(overrides)

    with hydra.initialize_config_dir(version_base=None, config_dir=str(sap.CONFIG_PATH)):
        cfg = hydra.compose('config', overrides=overrides_hydra)

        print(OmegaConf.to_container(cfg, resolve=True))

        # TODO: add a comment here what is happening?
        util.deserialize.set_ids_from_overrides(cfg, overrides)

        # TODO: Check if dataset was overriden if for instance additional labels
        # have been added swap out dataset config to access that data
        if util.deserialize.is_dataset_cfg_overridden(cfg.dataset.id):
            path = sap.OVERRIDE_CONFIG_DATASET_PATH / f'{cfg.dataset.id}.yaml'
            cfg.dataset = get_dataset_cfg_from_path(path)

        try:
            return ActiveMlConfig.model_validate(cfg)
        except ValidationError as e:
            logging.error(f"Could not parse hydra configuration as ActiveMlConfig with error: {e}")
            raise


def request_query(
        cfg: ActiveMlConfig,
        session_cfg: SessionConfig,
        X: np.ndarray,
) -> Batch:
    y = _load_or_init_annotations(X, cfg.dataset)
    query_func, clf = _setup_query(cfg, session_cfg)

    # Only fit and query on the samples not marked as outliers
    X_cand, y_cand, mapping = _filter_outliers(X, y)

    if clf is not None:
        print("Fitting the classifier")
        # TODO can fitting the classifier fail?
        # TODO filter out y that appear less then 2 times.
        # Some classifiers need at least 2 samples per class to train properly
        clf.fit(X_cand, y_cand)

        # TODO show how often class appeared
        # Only update when refitting?
        # unique_values, counts = np.unique(y_cand, return_counts=True)
        # for val, count in zip(unique_values, counts):
        #     logging.info(f'{val}: {count}')

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

    lenght = len(query_indices)
    batch_state = Batch(
        emb_indices=query_indices,
        class_probas=class_probas.tolist(),
        progress=0,
        annotations=[None] * lenght,
        start_times=[None] * lenght,
        end_times=[None] * lenght
    )
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
        data_path = sap.ROOT_PATH / data_path

    # TODO:: Check type and cast, it has to be an EmbeddingBadeAdapter here
    adapter = embedding_cfg.definition.instantiate()

    X, file_paths = adapter.compute_embeddings(data_path, progress_func)

    file_paths_str = _normalize_and_validate_paths(file_paths, X)

    # Unique key
    cache_key = f"{dataset_id}_{embedding_cfg.id}"
    cache_path = sap.EMBEDDINGS_CACHE_PATH / f"{cache_key}.npz"  # Use .npz to store multiple arrays

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


# TODO rename to update json_annotations
def completed_batch(dataset_id: str, batch: Batch, embedding_id: str) -> int:
    json_file_path = sap.ANNOTATED_PATH / f'{dataset_id}.json'
    print("completed batch")

    # Get existing annotations
    annotations: list[Annotation] = _deserialize_annotations(json_file_path)
    file_paths = get_file_paths(dataset_id, embedding_id, batch.emb_indices)

    new_annotations = [
        Annotation(
            embedding_idx=emb_idx,
            file_name=f_path,
            label=label,
            start_time=start,
            end_time=end
        )
        for emb_idx, f_path, label, start, end in zip(
            batch.emb_indices, file_paths, batch.annotations, batch.start_times, batch.end_times
        )
        if label != MISSING_LABEL_MARKER
    ]

    updated_annotations = annotations + new_annotations

    # Override annotations
    _serialize_annotations(json_file_path, updated_annotations)

    num_annotated = len(updated_annotations)
    return num_annotated


def get_num_annotated(dataset_id: str) -> int:
    json_file_path = sap.ANNOTATED_PATH / f'{dataset_id}.json'
    return len(_deserialize_annotations(json_file_path))


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
        # TODO use estimator to have more accurate terminology
        logging.warning("Cannot auto complete as there is not estimator selected!")
        return

    # TODO there is some repeated code here.
    random_state = np.random.RandomState(cfg.random_seed)
    estimator = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    X_cand, y_cand, _ = _filter_outliers(X, y)
    # TODO clf or estimator?
    clf = estimator
    clf.fit(X_cand, y_cand)

    X_missing, _, mapping = _filter_out_annotated(X, y)
    class_probas = clf.predict_proba(X=X_missing)  # shape (num_samples * num_labels)

    top_idxes = np.argmax(class_probas, axis=1)
    top_classes = clf.classes_[top_idxes]
    # Select top proba from each row
    top_probas = class_probas[np.arange(class_probas.shape[0]), top_idxes]

    # Select samples that are above the threshold probability
    is_threshold = (top_probas > threshold)
    emb_indices = mapping[is_threshold]
    selected_classes = top_classes[is_threshold]
    selected_probas = top_probas[is_threshold]

    # Get file paths for embedding indices
    selected_file_paths = get_file_paths(
        cfg.dataset.id,
        cfg.embedding.id,
        emd_indices=emb_indices
    )

    automated_annots = [
        AutomatedAnnotation(
            embedding_idx=int(emb_idx),
            label=label,
            file_name=f_path,
            confidence=float(proba),
        )
        for emb_idx, label, proba, f_path
        in zip(emb_indices, selected_classes, selected_probas, selected_file_paths)
    ]

    if sort_by_proba:
        automated_annots.sort(key=lambda ann: ann.confidence, reverse=True)

    json_file_path = sap.ANNOTATED_PATH / f'{cfg.dataset.id}.json'
    manual_annots = _deserialize_annotations(json_file_path)

    json_store_path = sap.ANNOTATED_PATH / f'{cfg.dataset.id}-automated.json'
    _serialize_annotations_with_keys(
        json_store_path,
        (manual_annots, automated_annots),
        ('manual', 'automated')
    )

    num_auto_annotated = len(automated_annots)
    num_total_annotated = num_auto_annotated + len(manual_annots)
    logging.info(f'{num_auto_annotated} samples have been automatically annoted @\n{json_store_path}')
    logging.info(f'In total annotated: {num_total_annotated}')

    # TODO Do we want to use automatically generated labels for further fitting?
    # total_annots = len(total_annots)
    # return total_annots


def save_partial_annotations(batch, dataset_id, embedding_id):
    for idx, val in enumerate(batch.annotations):
        # Put samples that have not been to missing so they come up again.
        if val is None:
            batch.annotations[idx] = MISSING_LABEL_MARKER
    num_annotated = completed_batch(dataset_id, batch, embedding_id)
    return num_annotated


def add_class(dataset_cfg, new_class_name: str) -> int:
    if new_class_name == '':
        logging.warning(f"Class name has to have at least lenght 1")
        # TODO I dont want to have dash in the api.
        raise dash.exceptions.PreventUpdate

    classes = dataset_cfg.classes

    if new_class_name in classes:
        logging.warning(f"Cannot add new class because {new_class_name} already exists.")
        raise dash.exceptions.PreventUpdate

    # TODO classes can be numbers in which case there is a different order.
    if util.utils.is_sorted(classes):
        insert_idx = bisect.bisect_left(classes, new_class_name)
        classes.insert(insert_idx, new_class_name)
    else:
        classes.append(new_class_name)
        insert_idx = len(classes) - 1

    sap.OVERRIDE_CONFIG_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    override_path = sap.OVERRIDE_CONFIG_DATASET_PATH / f'{dataset_cfg.id}.yaml'
    OmegaConf.save(config=dataset_cfg, f=override_path)

    # Invalidate cache. Force new composing when called next time.
    compose_config.cache_clear()

    logging.info(f'insert idx: {insert_idx}')
    return insert_idx
# endregion


# TODO put this stuff into utils package?
def _load_or_init_annotations(
        X: np.ndarray,  # TODO Should not take X but rather num_of_samples
        dataset_cfg: DatasetConfig
) -> np.ndarray:
    """Load existing labels or initialize with missing labels."""
    json_file_path = sap.ANNOTATED_PATH / f'{dataset_cfg.id}.json'
    num_samples = len(X)
    # TODO Performance. Dont repeat this computation
    max_label_name_len = max(
        len(s)
        for s in dataset_cfg.classes + [DISCARD_MARKER, MISSING_LABEL_MARKER]
    )

    # TODO for performance maybe it can be better to use ascci string with dtype=S
    # But then there is a disconnect.
    y = np.full(num_samples, MISSING_LABEL_MARKER, dtype=f'U{max_label_name_len}')

    if json_file_path.exists():
        y = _load_labels_as_np(y, json_file_path)

    return y


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


def _serialize_annotations_with_keys(
    path: Path,
    data: Sequence[Sequence[Annotation]],
    keys: Sequence[str]
):
    payload = {
        key: [asdict(obj) for obj in group]
        for key, group in zip(keys, data)
    }

    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)


def _load_labels_as_np(y: np.ndarray, json_path: Path) -> np.ndarray:
    """Load labels from a JSON file and return as a numpy array."""
    with json_path.open('r') as f:
        annotations = json.load(f)

        # TODO check if there is still some missing labels.
        # Else there is nothing more to label.
        for ann in annotations:
            idx = ann['embedding_idx']
            y[idx] = ann['label']

    return y


def _estimator_accepts_random(est_cls) -> bool:
    sig = inspect.signature(est_cls.__init__)
    return "random_state" in sig.parameters


# TODO bad name it should be filter_discard_samples
def _filter_outliers(X: npt.NDArray[np.number], y: npt.NDArray[np.number]):
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


# TODO will this be used for estmiators aswell?
def _build_activeml_classifier(
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        random_state: np.random.RandomState
) -> SkactivemlClassifier:
    classes = dataset_cfg.classes
    # n_classes = len(dataset_cfg.classes)
    # classes = np.arange(n_classes)

    # TODO rename to Estimator?
    est_cls = model_cfg.definition.target_

    kwargs = {}
    if _estimator_accepts_random(est_cls):
        kwargs['random_state'] = random_state

    est = model_cfg.definition.instantiate(**kwargs)

    if isinstance(est, SkactivemlClassifier):
        # Classifier is already wrapped aka supports missing labels
        # TODO missing_label wont have correct value.
        return est
    elif isinstance(est, sklearn.base.ClassifierMixin):
        wrapped_est = SklearnClassifier(
            estimator=est,
            classes=classes,
            random_state=random_state,
            missing_label=MISSING_LABEL_MARKER,
        )
        return wrapped_est
    else:
        raise RuntimeError(f"Estimator is not a sklearn ClassifierMixin")


# TODO can use from skactiveml.utils import call_func instead?
def _filter_kwargs(func: Callable, **kwargs) -> Callable:
    params = inspect.signature(func).parameters
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
        estimator = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

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
            missing_label=MISSING_LABEL_MARKER,
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
            full_p = sap.ROOT_PATH / p
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


def get_file_paths(
        dataset_id: str,
        embedding_id: str,
        emd_indices: list[int]
) -> np.ndarray[np.str_] | np.str_:
    cache_key = f'{dataset_id}_{embedding_id}'
    cache_path = sap.EMBEDDINGS_CACHE_PATH / f"{cache_key}.npz"

    if not cache_path.exists():
        raise RuntimeError(f"Cannot get embedding at path: {cache_path}! \nEmbedding should exists already")

    with np.load(str(cache_path), mmap_mode='r') as data:
        file_paths = data['file_paths']
        test = file_paths[emd_indices]
        return test


def undo_annots_and_restore_batch(cfg: ActiveMlConfig, num_undo: int) -> tuple[Batch | None, int]:
    # Assumes annotations are stored in json in the same order they were made.
    json_file_path = sap.ANNOTATED_PATH / f'{cfg.dataset.id}.json'
    annotations = _deserialize_annotations(json_file_path)

    # Check there is enough annotations made to go back that far.
    num_annotations = len(annotations)
    if num_undo > num_annotations:
        if num_annotations == 0:
            return None, 0

        num_undo = num_annotations

    write_back, reconstruct = annotations[:-num_undo], annotations[-num_undo:]

    labels, emb_idxes, starts, ends = [], [], [], []
    for annot in reconstruct:
        labels.append(annot.label)
        emb_idxes.append(annot.embedding_idx)
        starts.append(annot.start_time)
        ends.append(annot.end_time)

    _serialize_annotations(json_file_path, write_back)

    model_cfg = cfg.model
    if model_cfg is None:
        # TODO use estimator to have more accurate terminology
        estimator = None
    else:
        random_state = np.random.RandomState(cfg.random_seed)
        estimator = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    if estimator is not None:
        X = load_embeddings(cfg.dataset.id, cfg.embedding.id)
        y = _load_or_init_annotations(X, cfg.dataset)
        X_cand, y_cand, _ = _filter_outliers(X, y)

        estimator.fit(X_cand, y_cand)
        class_probas = estimator.predict_proba(X[emb_idxes])
    else:
        class_probas = np.empty(0)

    # Restored Batch
    return (
        Batch(
            emb_indices=emb_idxes,
            annotations=labels,
            class_probas=class_probas.tolist(),
            progress=num_undo,
            start_times=starts,
            end_times=ends,
        ),
        len(write_back)  # TODo Human annotations + automatic annotations.
    )
