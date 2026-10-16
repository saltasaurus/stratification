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

    def __init__(self, nin: int, nonlin=True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # "Tensor"
        self.b = Value(0) # Scalar
        self.nonlin = nonlin

    def __call__(self, x) -> Any:
        # w * x + b
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activation.relu() if self.nonlin else activation

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:

    def __init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
