import numpy as np


def as_tensaur(data) -> "Tensaur":
    return data if isinstance(data, Tensaur) else Tensaur(data)


class Tensaur:
    def __init__(self, data, _children=None, _op="", _label=""):
        self.data = np.asarray(data, dtype=float)

        self.grad = np.zeros_like(self.data)
        self._op = _op
        self._label = _label

        if _children is None:
            _children = ()
        self._prev = set(_children)

        self._backward = lambda: None

    def __add__(self, other: "Tensaur") -> "Tensaur":
        other = as_tensaur(other)

        out = Tensaur(self.data + other.data, (self, other))

        def _backward():
            # Self and other can have different shapes than out, so we must aggregate, not just broadcast the gradients
            self.grad += out.grad.sum() if out.grad.shape != self.grad.shape else out.grad
            other.grad += out.grad.sum() if out.grad.shape != other.grad.shape else out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Tensaur") -> "Tensaur":
        other = as_tensaur(other)

        out = Tensaur(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.grad, dtype=float)
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
