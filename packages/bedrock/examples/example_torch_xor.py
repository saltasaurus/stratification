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

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(3000):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()

        with torch.no_grad():
            optimizer.step()

        if step % 500 == 0:
            print(step, loss.item())

    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        print(probabilities)
        assert torch.equal((probabilities >= 0.5).float(), y)


if __name__ == "__main__":
    main()
