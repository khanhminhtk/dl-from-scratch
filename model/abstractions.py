from abc import ABC, abstractmethod

class ALinearModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit the linear model to the data."""
        pass

class AActivate_function(ABC):
    @abstractmethod
    def activate(self, X):
        """Apply the activation function."""
        pass

    @abstractmethod
    def derivative(self, X):
        """Compute the derivative of the activation function."""
        pass    

class AMultiLayerPerceptron(ABC):
    @abstractmethod
    def forward(self, X):
        """Forward pass through the MLP."""
        pass

    @abstractmethod
    def backward(self, y_true, grad_output=None):
        """Backward pass through the MLP."""
        pass

class AOptimizer(ABC):
    @abstractmethod
    def step(self):
        """Perform a single optimization step."""
        pass

    @abstractmethod
    def zero_grad(self):
        """Reset gradients for all parameters."""
        pass

class ASelfAttention(ABC):
    @abstractmethod
    def forward(self, X):
        """Forward pass through the self-attention mechanism."""
        pass

    # @abstractmethod
    # def backward(self, grad_output):
    #     """Backward pass through the self-attention mechanism."""
    #     pass