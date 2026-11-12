from bedrock.autograd import Tensaur
from bedrock.loss import bce_mean
from bedrock.nn import Linear
from bedrock.optim import Adam
from bedrock.random import manual_seed

MODEL_VERSION = "xor-v1"
manual_seed(42)


def predict(features: list[float]) -> float:

    t = Tensaur([features], requires_grad=False)

    model = Linear(1, 2, "xavier")

    out = model(t)

    return float(out.data.item())

    
