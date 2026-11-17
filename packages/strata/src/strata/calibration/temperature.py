from bedrock.autograd import Tensaur
from bedrock.loss import bce_mean
from bedrock.optim import Adam


def fit_temperature(logits, y, epochs=300, lr=0.01) -> float:
    log_T = Tensaur([0.0], requires_grad=True)
    opt = Adam([log_T], lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        T = log_T.exp()
        probs = (logits / T).sigmoid()
        loss = bce_mean(probs, y)
        loss.backward()
        opt.step()

    return float(log_T.exp().data.item())
