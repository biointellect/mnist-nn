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
            self.weights.append(rng.random((next, current), np.float64))
            self.biases.append(rng.random((next, 1), np.float64))


    def activation(self, values):
        """Activation function: ReLU"""
        return np.maximum(values, 0)


    def forward(self, input):
        """Forward propegate input through the network"""

        # Set first layer (input layer)
        self.layers[0] = input
        for i in range(len(self.layers) - 1):

            # 1. Multiply: weights @ input
            step_one = self.weights[i] @ self.layers[i]
            print("\nOne:\n", step_one)

            # 2. Add: biases
            step_two = step_one + self.biases[i]
            print("Two:\n", step_two)

            # 3. Activate: activation function
            step_three = self.activation(step_two)
            print("Three:\n", step_three)

            # 4. Save 
            self.layers[i + 1] = step_three

            # 4. Repeat until the output layer

        print("=====================")
        print(self.layers)



class Neuron:
    def __init__(self) -> None:
        pass

