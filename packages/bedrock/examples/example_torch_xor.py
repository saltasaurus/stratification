"""
An example showing a Torch MLP training on a non-linear problem (XOR).
This example is used to demonstrate the training loop so that Tensaur
can be correctly compared against.
"""

import torch
from torch import nn

torch.manual_seed(0)


def main():
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # model = nn.Sequential(
    #     nn.Linear(2, 8),
    #     nn.Tanh(),
    #     nn.Linear(8, 1),
    # )

    W1 = (torch.randn(2, 8) * 0.5).requires_grad_()
    b1 = torch.zeros(8, requires_grad=True)
    W2 = (torch.randn(8, 1) * 0.5).requires_grad_()
    b2 = torch.zeros(1, requires_grad=True)
    parameters = [W1, b1, W2, b2]

    def forward(x):
        hidden = torch.tanh(x @ W1 + b1)
        return hidden @ W2 + b2

    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(3000):
        for parameter in parameters:
            parameter.grad = None
        logits = forward(X)
        loss = loss_fn(logits, y)
        loss.backward()
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter -= parameter.grad * 0.1

        if step % 500 == 0:
            print(step, loss.item())

    with torch.no_grad():
        probabilities = forward(X).sigmoid()
        print(probabilities)
        assert torch.equal((probabilities >= 0.5).float(), y)


if __name__ == "__main__":
    main()
