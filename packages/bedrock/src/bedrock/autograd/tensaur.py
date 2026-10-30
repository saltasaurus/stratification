import numpy as np


def as_tensaur(data) -> "Tensaur":
    return data if isinstance(data, Tensaur) else Tensaur(data)


def collapse(x: np.ndarray, shape: tuple):
    """ """

    if x.shape == shape:
        return x

    while len(x.shape) > len(shape):
        x = x.sum(axis=0)

    for dim, sz in enumerate(shape):
        if sz == 1 and x.shape[dim] != 1:
            x = x.sum(axis=dim, keepdims=True)

    return x


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

    def __add__(self, other) -> "Tensaur":
        other = as_tensaur(other)

        out = Tensaur(self.data + other.data, (self, other), _op="add")

        def _backward():
            # Self and other can have different shapes than out, so we must aggregate, not just broadcast the gradients
            self.grad += collapse(out.grad, self.grad.shape)
            other.grad += collapse(out.grad, other.grad.shape)

        out._backward = _backward

        return out

    def __mul__(self, other) -> "Tensaur":
        other = as_tensaur(other)

        out = Tensaur(self.data * other.data, (self, other), _op="mul")

        def _backward():
            self.grad += collapse(other.data * out.grad, self.grad.shape)
            other.grad += collapse(self.data * out.grad, other.grad.shape)

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = as_tensaur(other)

        out = Tensaur(self.data @ other.data, (self, other), _op="matmul")

        def _backward():
            self.grad += collapse(out.grad @ other.data.T, self.grad.shape)
            other.grad += collapse(self.data.T @ out.grad, other.grad.shape)

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensaur(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> "Tensaur":  # -self
        return self * -1

    def __radd__(self, other) -> "Tensaur":  # other + self
        return self + other

    def __sub__(self, other) -> "Tensaur":  # self - other
        return self + (-other)

    def __rsub__(self, other) -> "Tensaur":  # other - self
        return other + (-self)

    def __rmul__(self, other) -> "Tensaur":  # other * self
        return self * other

    def __truediv__(self, other) -> "Tensaur":  # self / other
        return self * other**-1

    def __rtruediv__(self, other) -> "Tensaur":  # other / self
        return other * self**-1

    def __repr__(self) -> str:
        return f"Tensaur(data={self.data}, grad={self.grad})"

    def sum(self) -> "Tensaur":
        out = Tensaur(self.data.sum(), (self,), _op="sum")

        def _backward():
            self.grad += collapse(np.ones_like(self.data) * out.grad, self.grad.shape)

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

    # Custom unary operators
    # def sigmoid(self) -> "Tensaur":

    #     s = 1 / (1 + np.exp(-self.data))
    #     out = Tensaur(s, (self,), "sigmoid")

    #     def _backward():
    #         self.grad += collapse(s * (1 - s) * out.grad, self.grad.shape)

    #     out._backward = _backward
    #     return out

    def sigmoid(self) -> "Tensaur":

        data = np.empty_like(self.data, dtype=float)

        # mask of non-negative scores
        pos_mask = self.data >= 0
        data[pos_mask] = 1 / (1 + np.exp(-self.data[pos_mask]))

        # negative scores only
        exp_z = np.exp(self.data[~pos_mask])
        data[~pos_mask] = exp_z / (1 + exp_z)

        out = Tensaur(data, (self,), "sigmoid")

        def _backward():
            self.grad += collapse(data * (1 - data) * out.grad, self.grad.shape)

        out._backward = _backward

        return out

    def log(self) -> "Tensaur":

        out = Tensaur(np.log(self.data), (self,), "ln")

        def _backward():
            self.grad += collapse(out.grad / self.data, self.grad.shape)

        out._backward = _backward
        return out

    def clamp(self, xmin, xmax) -> "Tensaur":

        out = Tensaur(np.clip(self.data, xmin, xmax), (self,), _op="clamp")

        def _backward():
            valid: np.ndarray = (self.data >= xmin) & (self.data <= xmax)
            self.grad += collapse(valid.astype(float) * out.grad, self.grad.shape)

        out._backward = _backward
        return out
