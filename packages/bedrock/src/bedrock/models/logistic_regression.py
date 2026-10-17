import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.5, epochs=1000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            # 1. forward: z, then p
            # 2. gradient: err = p - y, then dw and db
            # 3. update w and b
            ...
        return self

    def predict_proba(self, X): ...
