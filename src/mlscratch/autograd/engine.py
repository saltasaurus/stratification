import math
import random
from typing import Any


class Value:

    def __init__(self, data, _children=None, _op='', label=''):
        self.data = data
        if _children is None:
            _children = ()
        self._prev = set(_children)
        self._op=_op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Wrap scalars into value
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other) -> 'Value':
        return -self + other

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Wrap scalars into value
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out 

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"

        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            # x ** k -> dx/dy = k*x**(k-1)
            self.grad += out.grad * (other * (self.data**(other-1)))

        out._backward = _backward

        return out
            

    def __truediv__(self, other): # self / over
        return self * other**-1

    
    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            # math.exp(x) is already calculated in out.data
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out 

    def sigmoid(self):
        t = 1 / (1 + math.exp(-self.data))
        out = Value(t, (self, ), 'sigmoid')

        def backward():
            self.grad += out.grad * (out.data * (1 - out.data))
        out._backward = backward

        return out

    def log(self):
        EPSILON = 1e-15
        x = min(self.data, 1 - EPSILON) 
        x = max(self.data, EPSILON)
        out = Value(math.log(x), (self, ), "log")

        def backward():
            self.grad += out.grad * self.data**-1
        out._backward = backward

        return out


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:

    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # ~Tensor
        self.b = Value(random.uniform(-1, 1)) # Scalar

    def __call__(self, x) -> Any:
        # w * x + b
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # add activation fn
        out = activation.sigmoid()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
                p.grad = 0.0

class Layer:
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        for p in self.parameters():
                p.grad = 0.0

class MLP:

    def __init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
                p.grad = 0.0
