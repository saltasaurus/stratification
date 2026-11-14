from bedrock.autograd import Tensaur
from bedrock.loss import bce_mean
from bedrock.nn import Linear
from bedrock.optim import Adam
from bedrock.random import manual_seed

MODEL_VERSION = "xor-v1"
manual_seed(42)

X = Tensaur(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    requires_grad=False,
)
y = Tensaur([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)

layer1, layer2 = Linear(2, 8, "xavier"), Linear(8, 1, "xavier")

parameters = layer1.parameters() + layer2.parameters()
optimizer = Adam(parameters, lr=0.01)


def forward(x: Tensaur):
    return layer2(layer1(x).tanh()).sigmoid()

for _ in range(3000):
    optimizer.zero_grad()

    probabilities = forward(X)
    loss = bce_mean(probabilities, y)
    loss.backward()
    optimizer.step()

def predict(features: list[float]) -> float:
    t = Tensaur([features], requires_grad=False)
    out = forward(t)
    return float(out.data.item())

    
