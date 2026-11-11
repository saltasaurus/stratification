import numpy as np
from bedrock.autograd.tensaur import Tensaur
from bedrock.loss.BCE import bce_mean
from bedrock.nn import Linear
from bedrock.optim import SGD, Adam
from bedrock.random import manual_seed

manual_seed(0)


X = Tensaur(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    requires_grad=False,
)
y = Tensaur([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)

layer1 = Linear(2, 8, "normal")
layer2 = Linear(8, 1, "normal")
parameters = layer1.parameters() + layer2.parameters()

# optimizer = SGD(parameters=parameters, lr=0.1, momentum=0.9)
optimizer = Adam(parameters, lr=0.01)


def forward(x: Tensaur):
    return layer2(layer1(x).tanh()).sigmoid()


for step in range(3000):
    optimizer.zero_grad()

    probabilities = forward(X)
    loss = bce_mean(probabilities, y)
    loss.backward()

    optimizer.step()

    if step % 250 == 0:
        print(step, loss.data.item())

probabilities = forward(X).data
print(probabilities)
np.testing.assert_array_equal(probabilities >= 0.5, y.data >= 0.5)
