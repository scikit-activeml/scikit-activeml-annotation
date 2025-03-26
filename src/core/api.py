from inspect import signature
from functools import partial
from typing import Callable

from hydra.utils import instantiate

from skactiveml.utils import MISSING_LABEL
from skactiveml.pool import SubSamplingWrapper
from skactiveml.classifier import SklearnClassifier
from skactiveml.base import (
    QueryStrategy,
    SkactivemlClassifier
)

from core.schema import *
from core.adapter import *

from util.deserialize import (
    parse_yaml_config_dir,
    parse_yaml_file
)
from paths import (
    DATA_CONFIG_PATH,
    ANNOTATED_PATH,
    QS_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    ADAPTER_CONFIG_PATH,
    DATASETS_PATH,
    CACHE_PATH
)


def _estimator_accepts_random(est_cls) -> bool:
    sig = inspect.signature(est_cls.__init__)
    return "random_state" in sig.parameters


# TODO what api do the pages need from the api?
# TODO will this be used for estmiators aswell?
def _build_activeml_classifier(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    random_state: np.random.RandomState
) -> SkactivemlClassifier:
    # TODO rename label names to classes to be more consistent with sklearn naming conv.
    # classes = dataset_cfg.label_names
    n_classes = len(dataset_cfg.label_names)
    classes = np.arange(n_classes)

    # TODO rename to Estimator?
    est_cls = model_cfg.definition['_target_']

    kwargs = {}
    if _estimator_accepts_random(est_cls):
        kwargs['random_state'] = random_state

    est = instantiate(model_cfg.definition, **kwargs)

    if isinstance(est, SklearnClassifier):
        # Classifier is already wrapped aka supports missing labels
        return est
    else:
        wrapped_est = SklearnClassifier(
            estimator=est,
            classes=classes,
            random_state=random_state,
            # missing_label=schema.MISSING_LABEL_STR
        )
        return wrapped_est


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
    qs: SubSamplingWrapper = SubSamplingWrapper(
        qs,
        max_candidates=session_cfg.max_candidates,
        random_state=random_state
    )

    # TODO separate query from fitting?
    query_func: Callable = _filter_kwargs(qs.query, batch_size=session_cfg.batch_size, clf=estimator, fit_clf=False,
                                          discriminator=estimator)
    return query_func, estimator


# region API
def get_dataset_config_options() -> list[DatasetConfig]:
    return parse_yaml_config_dir(DATA_CONFIG_PATH)


def get_qs_config_options() -> list[QueryStrategyConfig]:
    return parse_yaml_config_dir(QS_CONFIG_PATH)


def get_model_config_options() -> list[ModelConfig]:
    return parse_yaml_config_dir(MODEL_CONFIG_PATH)


def get_adapter_config_options() -> list[AdapterConfig]:
    return parse_yaml_config_dir(ADAPTER_CONFIG_PATH)


def get_query_cfg_from_id(query_id) -> QueryStrategyConfig:
    return parse_yaml_file(QS_CONFIG_PATH / f'{query_id}.yaml')


def _filter_outliers(X, y):
    mask = np.isfinite(y) | np.isnan(y)  # np.isfinite(np.nan) == False
    X_filtered = X[mask]
    y_filtered = y[mask]
    mapping = np.arange(len(X))[mask]
    return X_filtered, y_filtered, mapping


def request_query(
        cfg: ActiveMlConfig,
        session_cfg: SessionConfig,
        X: np.ndarray,
        file_names: list[str],
) -> Batch:
    y = _load_or_init_annotations(X, file_names, cfg.dataset.id)
    query_func, clf = _setup_query(cfg, session_cfg)

    print("shape of X in request_query")
    print(X.shape)
    print(y.shape)

    # Only fit and query on the samples not marked as outliers
    X_cand, y_cand, mapping = _filter_outliers(X, y)

    # TODO fitting classifier here might negate Subsampling QS wrapper
    if clf is not None:
        print("Fitting the classifier")
        clf.fit(X_cand, y_cand)

    print("Querying the active ML model ...")

    query_indices_cand = query_func(X=X_cand, y=y_cand)
    # Map back to original indices.
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
        class_probas = clf.predict_proba(query_samples)

    count_labeled_samples = np.sum(~np.isnan(y))
    total = X.shape[0]

    batch_state = Batch(
        indices=query_indices,
        class_probas=class_probas.tolist(),
        progress=0,
        annotations=[None] * len(query_indices)
    )
    print("\nNew Batch decided")
    print(f"Labaled/total: {count_labeled_samples} / {total}")
    return batch_state


# TODO cleanup make api clean.
# It should only be the interface for the pages. Not internal logic.
def get_or_compute_embeddings(
        dataset_cfg: DatasetConfig,
        adapter_cfg: AdapterConfig
) -> tuple[np.ndarray, list[str]]:
    """
    Resolve the data_path path, check/load cache if enabled, call compute_features,
    and cache the result if needed.
    """
    dataset_id = dataset_cfg.id
    data_path = dataset_cfg.data_path

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = DATASETS_PATH / data_path

    # Unique key
    cache_key = f"{dataset_id}_{adapter_cfg.id}"
    print(f"cache key: {cache_key}")

    cache_path = Path(str(CACHE_PATH)) / f"{cache_key}.npz"  # Use .npz to store multiple arrays

    if cache_path.exists():
        print(f"Cache hit. Loading cached features from {cache_path}")
        # Load both the feature matrix and the file names from the .npz cache
        with np.load(str(cache_path)) as data:
            X = data['X']
            file_paths = data['file_paths'].tolist()  # Convert to a list if necessary
        return X, file_paths
    else:
        import timeit

        print(f"Cache miss. Computing feature matrix and caching using adapter {adapter_cfg.id}...")

        # TODO Use of definition is not consistent.
        print("Start embedding ...")
        adapter: BaseAdapter = instantiate(adapter_cfg.definition)
        print("Selected adapter:", type(adapter))

        start_time = timeit.default_timer()
        X, file_paths = adapter.compute_embeddings(data_path)
        elapsed_time = timeit.default_timer() - start_time
        print(f"Embedding completed in: {elapsed_time:.2f} seconds")

        # Cache both the feature matrix and the file names in the .npz file
        np.savez(str(cache_path), X=X, file_paths=file_paths)

    return X, file_paths


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
