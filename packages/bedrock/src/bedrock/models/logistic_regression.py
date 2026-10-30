import numpy as np

from bedrock.autograd.tensaur import Tensaur
from bedrock.loss import bce_mean


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
    def __init__(self, lr=0.5, epochs=1000, l2=0.1):
        self.lr, self.epochs = lr, epochs
        self.l2reg = l2

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
        self.w, self.b = Tensaur(np.zeros((d, 1))), Tensaur(np.zeros((1, 1)))

        for epoch in range(self.epochs):
            self.w.zero_grad()
            self.b.zero_grad()

            logits = X @ self.w + self.b
            probs = logits.sigmoid()
            reg_loss = (self.l2reg / 2) * (self.w * self.w).sum()
            loss = bce_mean(probs, y) + reg_loss
            loss.backward()

            self.w.data -= self.lr * self.w.grad
            self.b.data -= self.lr * self.b.grad

            if epoch % 100 == 0:
                print(f"Epoch {epoch} loss: {loss.data}")

        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, size=(30, 2)),
            rng.normal(loc=1.0, size=(30, 2)),
        ]
    )
    y = np.array([[0.0]] * 30 + [[1.0]] * 30)
    model = LogisticRegression(lr=0.5, epochs=1000, l2=0.5)
    model.fit_tensaur(Tensaur(X), Tensaur(y))
    w = np.asarray(model.w.data, dtype=float)
    b = np.asarray(model.b.data, dtype=float)
    p = sigmoid(X @ w + b)
    acc = float(np.mean((p >= 0.5) == y))
    print(f"final acc={acc:.2f}\nw={w}\nb={b}")
