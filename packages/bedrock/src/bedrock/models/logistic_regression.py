import numpy as np

from bedrock.autograd.tensaur import Tensaur
from bedrock.loss import bce_mean


class LogisticRegression:
    def __init__(self, lr=0.5, epochs=1000, l2=0.1):
        self.lr, self.epochs = lr, epochs
        self.l2reg = l2

    def fit(self, X: Tensaur, y: Tensaur):
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

    def predict(self, X: Tensaur):
        return (X @ self.w + self.b).sigmoid()
