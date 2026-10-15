import math

from nn.modules import Neuron


def sigmoid(x):

    return 1 / (1 + math.exp(-x))

def main():
    print("Hello from micrograd!")

    n = Neuron(5)

    print(sigmoid(2))


if __name__ == "__main__":
    main()
