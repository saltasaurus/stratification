from bedrock.autograd.tensaur import Tensaur


class SGD:
    def __init__(self, parameters: list[Tensaur], lr=0.01):
        self.params = list(parameters)  # Create stable list over potential generator
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
