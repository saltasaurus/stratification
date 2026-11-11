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

class Adam:
    def __init__(self, parameters: list[Tensaur], lr=0.001, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(parameters)
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.EPS = eps
        self._step = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        self._step += 1
        bc1 = (1 - self.beta_1 ** self._step) 
        bc2 = (1 - self.beta_2 ** self._step) 
        for p, m, v in zip(self.params, self.m, self.v, strict=False):
            g = p.grad
            # Update biased first moment estimate
            m *= self.beta_1 # 
            m += (1 - self.beta_1) * g
            # Update biased second moment estimate
            v *= self.beta_2
            v += ((1 - self.beta_2) * g**2)

            # Compute bias-corrected first & second moment estimate
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.EPS)
