import numpy as np

class ActivationFunction:
    """Base class for activation functions."""
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")
    def backward(self, x):
        raise NotImplementedError("Backward method not implemented.")
    

class sigmoid(ActivationFunction):
    """Sigmoid activation function."""
    def forward(self, x):
        """
        Compute the sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying sigmoid function.
        """
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, x):
        """
        Compute the derivative of the sigmoid function.
        Args:
            x (np.ndarray): Input array.
        Returns:
            np.ndarray: Derivative of the sigmoid function.
        """
        return self.out * (1 - self.out)

class relu(ActivationFunction):
    """Relu activation function."""

    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, x):
        grad = np.where(self.out > 0, 1, 0)
        return grad