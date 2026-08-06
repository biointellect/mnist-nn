# Neural network manipulation
import numpy as np


class Network:
    def __init__(self, dimensions: list[int]) -> None:
        """Initialize the neural network with dimensions, a list of number of neurons in each layer, including the input and output layers."""
        # Init the layers as a list of 2d arrays
        self.layers = []
        self.dims = dimensions

        for dim in self.dims:
            self.layers.append(np.zeros(dim, np.float64))

        # Init the weights as a list of 2d arrays and init the biases as a list of 1d arrays
        self.weights = []
        self.biases = []
        rng = np.random.default_rng()

        for i in range(len(self.dims) - 1):
            current = self.dims[i]
            next = self.dims[i + 1]
            self.weights.append(rng.random((current, next), np.float64))
            self.biases.append(rng.random(current, np.float64))


    def activation(self):
        raise NotImplementedError("activation() must be implemented")


    def forward(self, input):
        """Forward propegate input through the network"""
        # 1. Multiply: weights . input

        # 2. Add: biases

        # 3. Activate: activation function

        # 4. Repeat until the output layer
        raise NotImplementedError("forward() must be implemented")



class Neuron:
    def __init__(self) -> None:
        pass

