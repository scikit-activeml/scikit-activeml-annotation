from io import BytesIO
import base64
from typing import TypeVar, TypeGuard
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from itertools import islice
import json
import logging
import inspect
import bisect
from dataclasses import Field, asdict 
from functools import partial, lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar, cast, Protocol
import dash.exceptions

import hydra
import pydantic
from omegaconf import OmegaConf

import numpy as np
import numpy.typing as npt

import sklearn
from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper

from skactiveml_annotation import util
from skactiveml_annotation.util import deserialize
import skactiveml_annotation.paths as sap

from skactiveml_annotation.core.schema import (
    ActiveMlConfig,
    AnnotationList,
    EmbeddingConfig,
    QueryStrategyConfig,
    ModelConfig,
    DatasetConfig,
    SessionConfig,
    Annotation,
    AutomatedAnnotation,
    Batch,
    MISSING_LABEL_MARKER,
    DISCARD_MARKER,
    HistoryIdx,
)

from skactiveml_annotation.core.shared_types import DashProgressFunc

QueryFunc = Callable[..., npt.NDArray[np.intp]]

class DataClassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


T = TypeVar("T")

def _not_none_type_narrowing(x: T | None) -> TypeGuard[T]:
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

def get_dataset_cfg_from_path(path: Path) -> DatasetConfig:
    return deserialize.parse_yaml_file(path, DatasetConfig)

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

        # TODO: add a comment here what is happening?
        deserialize.set_ids_from_overrides(cfg, overrides)

        # TODO: Check if dataset was overriden if for instance additional labels
        # have been added swap out dataset config to access that data
        if deserialize.is_dataset_cfg_overridden(cfg.dataset.id):
            path = sap.OVERRIDE_CONFIG_DATASET_PATH / f'{cfg.dataset.id}.yaml'
            cfg.dataset = get_dataset_cfg_from_path(path)

        try:
            return ActiveMlConfig.model_validate(cfg)
        except pydantic.ValidationError as e:
            logging.error(f"Could not parse hydra configuration as ActiveMlConfig with error: {e}")
            raise


def _get_sklearn_classes(clf: SkactivemlClassifier) -> list[str]:
    """
    Extracts the classes_ attribute from a SkactivemlClassifier and returns it as a list of strings.
    
    Parameters:
        clf (SkactivemlClassifier): The classifier from which to extract classes.
    
    Returns:
        List[str]: List of class names. Returns [""] if extraction fails.
    """
    try:
        raw_classes = getattr(clf, "classes_", None)
        if raw_classes is None:
            raise AttributeError("clf.classes_ is None (model not fitted?)")

        classes_sklearn = raw_classes.tolist()
        if not isinstance(classes_sklearn[0], str):
            logging.warning("Is it not strings?")
        classes_sklearn = cast(list[str], classes_sklearn)

    except Exception as e:
        logging.error("Failed to extract clf.classes_: %s", e, exc_info=True)
        classes_sklearn = [""]

    return classes_sklearn


def request_query(
    cfg: ActiveMlConfig,
    session_cfg: SessionConfig,
    X: np.ndarray,
    filter_out_emb_indices: list[int]
) -> tuple[Batch, AnnotationList]:
    y = _load_or_init_annotations(X, cfg.dataset)

    # INFO: Dont query on these samples in filter_out_emb_indices by marking them as discarded
    y[filter_out_emb_indices] = DISCARD_MARKER

    query_func, clf = _setup_query(cfg, session_cfg)

    # Only fit and query on the samples not marked as discarded
    X_cand, y_cand, mapping = _filter_outliers(X, y)

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

    logging.info("Querying the active ML model ...")

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
    # TODO: Should it not alwayls return a numpy array with ints?
    if not isinstance(query_indices[0], int):
        query_indices = [int(x) for x in query_indices]

    query_samples = X[query_indices]

    # TODO: clf could not have predict_proba

    # From doc prodict Proba always returns 
    # Probabilites P of : array-like of shape (n_samples, classes)
    # get turned into list[list[float]] by to_list()
    class_probas = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        # npt.NDArray[np.float64], 
        clf.predict_proba(query_samples)
    )
    class_probas_list = cast(list[list[float]], class_probas.tolist())

    classes_sklearn = _get_sklearn_classes(clf)

    # Possibly restore annotations that have been previously skipped
    # TODO what are all the places you are getting annotaitons
    # TODO write helper for this? Restore annotations or something
    file_paths = get_file_paths(cfg.dataset.id, cfg.embedding.id, query_indices)
    annotations_data = _deserialize_annotations(cfg.dataset.id)
    # file_paths are the keys
    annotations_list = AnnotationList(
        annotations = [
            annotations_data.get(f_path, None) for f_path in file_paths
        ]
    )

    return (
        Batch(
            emb_indices=query_indices,
            class_probas=class_probas_list,
            classes_sklearn=classes_sklearn,
            progress=0,
        ),
        annotations_list
    )

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


# TODO rename to update json_annotations
def completed_batch(
    dataset_id: str, 
    embedding_id: str, 
    new_annotations: list[Annotation],
    batch: Batch # TODO really only the emd_indices are needed
):
    print("\ncompleted batch")

    file_paths = get_file_paths(dataset_id, embedding_id, batch.emb_indices)

    update_annotations(
        dataset_id, 
        file_paths,
        new_annotations
    )

    # Assumes the idx is on the first of the current batch
    # Put the idx on the last element of the batch
    # TODO: No longer increment index to last position
    increment_global_history_idx(dataset_id, len(new_annotations) - 1)
    print("Increment history_idx to: ", get_global_history_idx(dataset_id))


# TODO: Not needed?
def get_num_annotated_ram(annotations: Iterable[Annotation | None]):
    # Samples are counted as annotated if they have a label or are discareded
    cnt = 0
    for annot in annotations:
        if annot is not None and annot.label != MISSING_LABEL_MARKER:
            cnt += 1
    return cnt


def get_num_annotated_not_skipped(dataset_id: str) -> int:
    annotations = _deserialize_annotations(dataset_id)

    return sum(
        (
            1
            for annot in annotations.values()
            if annot.label != MISSING_LABEL_MARKER
        )
    )


def get_num_annotated(dataset_id: str) -> int:
    return len(_deserialize_annotations(dataset_id))


def get_total_num_samples(dataset_id: str, embedding_id: str) -> int:
    return len(load_embeddings(dataset_id, embedding_id))


def auto_annotate(
    X: np.ndarray, # TODO: type?
    cfg: ActiveMlConfig,
    threshold: float,
    sort_by_proba: bool = True
):
    y = _load_or_init_annotations(X, cfg.dataset)

    model_cfg = cfg.model
    if model_cfg is None:
        # TODO use estimator to have more accurate terminology
        logging.warning("Cannot auto complete as there is no estimator selected!")
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

    assert clf.classes_ is not None
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
            file_path=f_path,
            confidence=float(proba),
        )
        for emb_idx, label, proba, f_path
        in zip(emb_indices, selected_classes, selected_probas, selected_file_paths)
    ]

    if sort_by_proba:
        automated_annots.sort(key=lambda ann: ann.confidence, reverse=True)

    manual_annots = list(_deserialize_annotations(cfg.dataset.id).values()) 

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


def save_partial_annotations(batch: Batch, dataset_id: str, embedding_id: str, annotations: list[Annotation | None]):
    annotated = list(filter(_not_none_type_narrowing, annotations)) 
    completed_batch(dataset_id, embedding_id, annotated, batch)


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


# TODO put this stuff into utils package?
def _load_or_init_annotations(
        X: np.ndarray,  # TODO Should not take X but rather num_of_samples
        dataset_cfg: DatasetConfig
) -> np.ndarray:
    """Load existing labels or initialize with missing labels."""
    num_samples = len(X)
    # TODO Performance. Dont repeat this computation
    max_label_name_len = max(
        len(s)
        for s in dataset_cfg.classes + [DISCARD_MARKER, MISSING_LABEL_MARKER]
    )

    # TODO for performance maybe it can be better to use ascci string with dtype=S
    # But then there is a disconnect.
    y = np.full(num_samples, MISSING_LABEL_MARKER, dtype=f'U{max_label_name_len}')

    # if json_file_path.exists():
    _load_labels_as_np(y, dataset_cfg.id)

    return y


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
                [(key, ann.model_dump()) for key, ann in annotations.items()]
            ),
            f,
            indent=4
        )

def update_annotations(
    dataset_id: str,
    file_paths: list[str],
    new_annotations: Sequence[Annotation | None],
    move_to_end_on_update: bool = True
): 
    annotations = _deserialize_annotations(dataset_id)
    # Get file_paths as they are the keys

    new_annotations_dict = OrderedDict(
        (f_path, annot) for f_path, annot in zip(file_paths, new_annotations)
        if annot is not None
    )

    # TODO
    # new_annotations_dict = {
    #     f_path: annot
    #     for f_path, annot in zip(file_paths, new_annotations) if annot is not None
    # }

    # TODO its always moved to end now, which should not always be the case

    # TODO: Dont move to end
    # if move_to_end_on_update:
    #     for key, item in new_annotations_dict.items():
    #         # INFO: Only move skipped samples to the end
    #         annotations[key] = item
    #         if item.label == MISSING_LABEL_MARKER:
    #             print("Move to end")
    #             annotations.move_to_end(key, last=True)
    # else:
        # annotations.update(new_annotations_dict)

    annotations.update(new_annotations_dict)

    _serialize_annotations(dataset_id, annotations)


def _serialize_annotations_with_keys(
    path: Path,
    data: Sequence[Sequence[DataClassInstance]],
    keys: Sequence[str]
):
    # TODO change serialization
    payload = {
        key: [asdict(obj) for obj in group]
        for key, group in zip(keys, data)
    }

    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)


def _load_labels_as_np(y: np.ndarray, dataset_id: str):
    """Load labels from a JSON file and return as a numpy array."""
    annotations = _deserialize_annotations(dataset_id)

    num_annotations = len(annotations)
    emb_indices = np.empty(num_annotations, dtype=int)
    # TODO Use label encoder earlier?
    labels = np.empty(num_annotations, dtype=object)

    for i, ann in enumerate(annotations.values()):
        emb_indices[i] = ann.embedding_idx
        labels[i] = ann.label
    
    y[emb_indices] = labels

    # for ann in annotations.values():
    #     idx = ann.embedding_idx
    #     y[idx] = ann.label


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

    # TODO: just pass key value pair here
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
            missing_label=MISSING_LABEL_MARKER,  # pyright: ignore[reportArgumentType]
        )
        return wrapped_est
    else:
        raise RuntimeError(f"Estimator is not a sklearn ClassifierMixin")


# TODO can use from skactiveml.utils import call_func instead?
def _filter_kwargs(func: QueryFunc, **kwargs) -> QueryFunc:
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
def _setup_query(cfg: ActiveMlConfig, session_cfg: SessionConfig) -> tuple[QueryFunc, SkactivemlClassifier]:
    random_state = np.random.RandomState(cfg.random_seed)

    model_cfg = cfg.model
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
            # from doc missing_label: scalar | str | np.nan | None,
            missing_label=MISSING_LABEL_MARKER,  # pyright: ignore[reportArgumentType]
        )

    # Dont fit classifier here to prevent fitting twice
    # TODO:: Does each one have fit_clf flag?
    query_func = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=estimator, fit_clf=False,
                                          discriminator=estimator)
    return query_func, estimator


def _normalize_and_validate_paths(
        file_paths: list[Path],
        X: np.ndarray
) -> list[str]:
    if len(file_paths) != len(X):
        raise RuntimeError(f'Amount of samples does not match amount of file paths!')

    file_paths_str = []
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
        print("[WARNING] absolute paths were provided. Results won't be easily shareable.")
    return file_paths_str


def get_one_file_path(
    dataset_id: str,
    embedding_id: str,
    emb_idx: int
) -> str:
    return get_file_paths(dataset_id, embedding_id, emb_idx)[0]


# TODO: this should convert to Path allready
# it should maybe just return a list[str]?
# My ui should not deal with numpy stuff
def get_file_paths(
    dataset_id: str,
    embedding_id: str,
    emd_indices: np.ndarray[tuple[int], np.dtype[np.intp]] | list[int] | int
) -> list[str]:
    cache_key = f'{dataset_id}_{embedding_id}'
    cache_path = sap.EMBEDDINGS_CACHE_PATH / f"{cache_key}.npz"

    if isinstance(emd_indices, int):
        emd_indices = [emd_indices]

    if not cache_path.exists():
        raise RuntimeError(f"Cannot get embedding at path: {cache_path}! \nEmbedding should exists already")

    with np.load(str(cache_path), mmap_mode='r') as data:
        file_paths = data['file_paths']
        # tolist() returns np.array if given a list
        return file_paths[emd_indices].tolist()


def get_global_history_idx(dataset_id: str) -> int | None:
    """
    Retrieve the history index for a given dataset ID.
    Returns None if the file dose not exist
    """
    path = sap.HISTORY_IDX / f"{dataset_id}.json"

    if not path.exists():
        return None

    # Read JSON from file
    content = path.read_text()
    model = HistoryIdx.model_validate_json(content)
    return model.idx


def set_global_history_idx(dataset_id: str, value: int) -> None:
    """
    Store (or update) the history index for a given dataset ID.
    Creates the directory if needed.
    """
    # print("SET GLOBAL IDX to: ", value)
    path = sap.HISTORY_IDX / f"{dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    model = HistoryIdx(idx=value)

    path.write_text(model.model_dump_json(indent=4))

def increment_global_history_idx(dataset_id: str, value: int):
    # print("INC IDX by:", value)
    current_idx = get_global_history_idx(dataset_id)
    assert current_idx is not None

    new_val = current_idx + value
    set_global_history_idx(dataset_id, new_val)
    
def restore_batch(
    cfg: ActiveMlConfig, 
    history_idx: int, 
    restore_forward: bool,
    num_restore: int
) -> tuple[Batch, AnnotationList]:
    # INFO: When restoring backwards it will try to restore num_restore samples
    # If there are not enough samples left to restore it will restore as much as it can
    # If it cant restore it will throw an error
    # Assumes annotations are stored in json in the same order they were made.
    print("\nRestore Batch")
    print("history idx:", history_idx)

    # INFO: History_idx is exclusive and wont be restored

    if restore_forward:
        start = history_idx + 1
        end = start + num_restore
    else:
        end = history_idx # exclusive
        # TODO: 
        start = max(0, end - num_restore) 

        num_restorable = end - start

        if num_restorable <= 0:
            logging.info(f"There is no samples left to restore backwards")
            raise RuntimeError()

        elif num_restorable < num_restore:
            logging.info(f"Can not restore backwards {num_restore}, only {num_restorable} will be restored")
        
    print("start:", start)
    print("end:", end, "(exclusive)")

    annotations_data = _deserialize_annotations(cfg.dataset.id)
    sliced = islice(annotations_data.values(), start, end)
    annotations = list(sliced) 

    print("len restored:")
    print(len(annotations))

    emb_idxes = [annot.embedding_idx for annot in annotations]
    
    model_cfg = cfg.model
    random_state = np.random.RandomState(cfg.random_seed)
    estimator = _build_activeml_classifier(model_cfg, cfg.dataset, random_state=random_state)

    X = load_embeddings(cfg.dataset.id, cfg.embedding.id)
    y = _load_or_init_annotations(X, cfg.dataset)
    X_cand, y_cand, _ = _filter_outliers(X, y)

    estimator.fit(X_cand, y_cand)
    class_probas = estimator.predict_proba(X[emb_idxes])

    #  TODO workarround typing 
    annotations = cast(list[Annotation | None], annotations)

    return (
        Batch(
            emb_indices=emb_idxes,
            class_probas=class_probas.tolist(),
            classes_sklearn=_get_sklearn_classes(estimator),
            progress=0 if restore_forward else len(emb_idxes) - 1
        ),
        AnnotationList(annotations=annotations)
    )


def file_buffer_to_inline_data_url(file_data_buffer: BytesIO, mime: str) -> str:
    b64_encoded_file_data = base64.b64encode(file_data_buffer.getvalue()).decode()
    return f"data:{mime};base64,{b64_encoded_file_data}"
