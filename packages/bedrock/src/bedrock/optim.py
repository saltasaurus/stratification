import numpy as np

from bedrock.autograd.tensaur import Tensaur


class SGD:
    def __init__(self, parameters: list[Tensaur], lr=0.01, momentum=0.0):
        self.params = list(parameters)  # Create stable list over potential generator
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        for p, velocity in zip(self.params, self.velocities, strict=True):
            # Must be in-place to keep self.velocities up to date
            velocity *= self.momentum
            velocity += p.grad
            p.data -= self.lr * velocity
