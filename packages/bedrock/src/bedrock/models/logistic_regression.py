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
            s = X @ self.w + self.b  # scores = features @ weights + bias
            p = sigmoid(s)  # probabilities = sigmoid(scores)
            err = p - y  # errors = probabilities - labels
            dw = X.T @ err / n  # grad weights = average(features * error)
            db = err.mean()  # grad bias = average(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


if __name__ == "__main__":
    X = np.array([[0.5], [1.0], [1.5], [2.5], [3.0], [3.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(epochs=100)

    model.fit(X, y)

    print(model.predict_proba(X))
    # decision boundary for single feature derived from the following:
    # Decision flips from 0<->1 at sigmoid(score)=0.5, which is when score=0, therefore
    # > score = weight * x + bias = 0
    # > x = -bias / weight
    print(-model.b / model.w[0])
