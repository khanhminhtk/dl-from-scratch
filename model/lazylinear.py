from model.abstractions import ALinearModel
from tqdm import tqdm
import torch

class LinearModel(ALinearModel):
    def __init__(self, input_dim, output_dim, mean_init, std_init, drive) -> None:
        self.weights = self._init_weights(input_dim, output_dim, mean_init, std_init).to(drive)
        self.bias = self._init_bias(output_dim, mean_init, std_init).to(drive)
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)

    def _init_weights(self, input_dim, output_dim, mean_init, std_init) -> torch.Tensor:
        weights = torch.normal(
            mean=mean_init,
            std=std_init,
            size=(input_dim, output_dim)
        )
        return weights
    
    def _init_bias(self, output_dim, mean_init, std_init) -> torch.Tensor:
        bias = torch.normal(
            mean=mean_init,
            std=std_init,
            size=(1, output_dim)
        )
        return bias
    
    def fit(self, X):
        return torch.mm(X, self.weights) + self.bias

    def zero_grad(self):
        self.grad_weights.zero_()
        self.grad_bias.zero_()

    def parameters(self):
        return [
            {"param": self.weights, "grad": self.grad_weights},
            {"param": self.bias, "grad": self.grad_bias}
        ]


# if __name__ == "__main__":
#     drive = "cuda" if torch.cuda.is_available() else "cpu" 
#     A = torch.randn(3, 4)
#     A = A.to(drive)
#     linear = LinearModel(input_dim=4, output_dim=2, mean_init=0.0, std_init=1.0, drive=drive)
#     output = linear.fit(A)
#     print(output)