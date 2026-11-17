import numpy as np


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    pred = probs >= 0.5
    conf = np.where(pred == 1, probs, 1 - probs)
    correct = pred == y

    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(conf, edges[1:-1], right=True)

    n = len(probs)
    score = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if not in_bin.any():
            continue

        acc = correct[in_bin].mean()
        avg_conf = conf[in_bin].mean()
        score += (in_bin.sum() / n) * abs(acc - avg_conf)

    return float(score)
