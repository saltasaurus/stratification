import numpy as np
from bedrock.autograd.tensaur import Tensaur
from bedrock.loss.BCE import bce_mean

rng = np.random.default_rng(0)

X = Tensaur(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    requires_grad=False,
)
y = Tensaur([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)

W1 = Tensaur(rng.normal(scale=0.5, size=(2, 8)))
b1 = Tensaur(np.zeros(8))
W2 = Tensaur(rng.normal(scale=0.5, size=(8, 1)))
b2 = Tensaur(np.zeros(1))
parameters = [W1, b1, W2, b2]


def forward(x):
    hidden = (x @ W1 + b1).tanh()
    return (hidden @ W2 + b2).sigmoid()


for step in range(3000):
    for parameter in parameters:
        parameter.grad.fill(0)

    probabilities = forward(X)
    loss = bce_mean(probabilities, y)
    loss.backward()

    for parameter in parameters:
        parameter.data -= 0.1 * parameter.grad

    if step % 500 == 0:
        print(step, loss.data.item())

probabilities = forward(X).data
print(probabilities)
np.testing.assert_array_equal(probabilities >= 0.5, y.data >= 0.5)
