import numpy as np

_rng = np.random.default_rng()

def manual_seed(seed: int) -> None:
    global _rng
    _rng = np.random.default_rng(seed)

def get_rng() -> np.random.Generator:
    return _rng