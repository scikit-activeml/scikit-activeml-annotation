import numpy as np

def start_labelling_session():
    raise NotImplementedError

def stop_labelling_session():
    raise NotImplementedError

def req_annotation(query_indices, n_classes: int, random_state: np.random.RandomState) -> np.ndarray:
    """
    Returns np array-like of shape (query_indices)
    """

    # TODO for now select random classes for each requested label.
    return random_state.randint(0, n_classes, size=len(query_indices))


def finish_label_session():
    # Notify UI that. There is not more labels to label.
    print("Not Implemented")

def retrain():
    raise NotImplementedError