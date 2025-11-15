from model.abstractions import AActivate_function
import torch

class ReluActivateFunction(AActivate_function):
    def activate(self, X: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.zeros_like(X), X)
    
    def derivative(self, X: torch.Tensor) -> torch.Tensor:
        return (X > 0).float()
    
class SigmoidActivateFunction(AActivate_function):
    def activate(self, X: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-X))
    
    def derivative(self, X: torch.Tensor) -> torch.Tensor:
        sig = self.activate(X)
        return sig * (1 - sig)
    
class TanhActivateFunction(AActivate_function):
    def activate(self, X: torch.Tensor) -> torch.Tensor:
        return (torch.exp(X) - torch.exp(-X)) / (torch.exp(X) + torch.exp(-X))
    
    def derivative(self, X: torch.Tensor) -> torch.Tensor:
        tanh_x = self.activate(X)
        return 1 - tanh_x ** 2
    
class softmaxActivateFunction(AActivate_function):
    def activate(self, X: torch.Tensor) -> torch.Tensor:
        exp_X = torch.exp(X - torch.max(X, dim=1, keepdim=True).values)
        return exp_X / torch.sum(exp_X, dim=1, keepdim=True)
    
    def derivative(self, X: torch.Tensor) -> torch.Tensor:
        softmax_x = self.activate(X)
        diag = torch.diag_embed(softmax_x)
        outer = softmax_x.unsqueeze(2) * softmax_x.unsqueeze(1)
        return diag - outer


# if __name__ == "__main__":
#     A = torch.tensor([[-1.0, 2.0, -3.0],
#                       [4.0, -5.0, 6.0]])
#     relu = ReluActivateFunction()
#     activated = relu.activate(A)
#     derivative = relu.derivative(A)
#     print("Activated:\n", activated)
#     print("Derivative:\n", derivative)
#     A = A.to("cuda")
#     activated = softmaxActivateFunction()
#     activated = activated.activate(A)
#     print("Activated:\n", activated)