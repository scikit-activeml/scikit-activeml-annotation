from dataclasses import dataclass, field

@dataclass
class SessionConfig:
    n_cycles: int = 10
    batch_size: int = 2
    max_candidates: int | float = 1000

def fetch_session_config() -> SessionConfig:
    # TODO actually request data from UI.
    # for now return dummy data
    return SessionConfig()


@dataclass
class SessionState:
    n_labeled: int = 0




    