import random
from typing import Any

from .engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin: int, act="relu") -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # "Tensor"
        self.b = Value(0)  # Scalar
        self.act = act

    def __call__(self, x) -> Any:
        # w * x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x, strict=False)), self.b)
        if self.act == "relu":
            return activation.relu()
        elif self.act == "tanh":
            return activation.tanh()
        elif self.act == "sigmoid":
            return activation.sigmoid()
        else:
            return activation  # linear

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{self.act} Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin: int, nout: int, act="relu", **kwargs) -> None:
        self.neurons = [Neuron(nin, act=act, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP:
    def __init__(self, nin: int, nouts: list[int], act="relu", out_act="linear") -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], act=act if i < len(nouts) - 1 else out_act) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
