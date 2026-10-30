import numpy as np

from bedrock.autograd.tensaur import Tensaur


def sigmoid(z: np.ndarray) -> np.ndarray:

    out = np.empty_like(z, dtype=float)

    # mask of non-negative scores
    pos_mask = z >= 0
    out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))

    # negative scores only
    exp_z = np.exp(z[~pos_mask])
    out[~pos_mask] = exp_z / (1 + exp_z)

    return out


class LogisticRegression:
    def __init__(self, lr=0.5, epochs=1000):
        self.lr, self.epochs = lr, epochs
        self.l2reg = 0.1

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            s = X @ self.w + self.b  # scores = features @ weights + bias
            p = sigmoid(s)  # probabilities = sigmoid(scores)
            err = p - y  # errors = probabilities - labels
            dw = X.T @ err / n  # grad weights = average(features * error)
            db = err.mean()  # grad bias = average(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def fit_tensaur(self, X: Tensaur, y: Tensaur):
        n, d = X.data.shape
        self.w, self.b = Tensaur(np.zeros(d)), 0.0

        for _ in range(self.epochs):
            s = X @ self.w + self.b
            p = sigmoid(s)
            err = (p - y) + (self.l2reg / 2) * (self.w * self.w).sum()
            dw = X.T @ err / n
            db = err.mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


if __name__ == "__main__":
    x = np.array([-1000, 0, 1000])

    print(sigmoid(x))
