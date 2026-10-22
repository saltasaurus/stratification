import numpy as np


def empirical_cdf(x, y) -> float:
    """x <= y for each value in y over the total length of x"""
    return np.searchsorted(np.sort(x), y, side="right") / len(x)


def ks_statistic(a, b) -> float:
    grid_pool = np.concat([a, b])
    return np.max(np.abs(empirical_cdf(a, grid_pool) - empirical_cdf(b, grid_pool)))
