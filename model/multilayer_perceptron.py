from typing import Optional

from model.abstractions import AMultiLayerPerceptron
from model.lazylinear import LinearModel
from model.activate_function import ReluActivateFunction, SigmoidActivateFunction, TanhActivateFunction, softmaxActivateFunction
from optimation.optimizers import SGD, SGDMomentum, Adam, RMSprop
import torch


class MultiLayerPerceptron(AMultiLayerPerceptron):
    def __init__(self, drive, input_dim, output_dim) -> None:
        self.drive = drive
        self.layer_1 = LinearModel(
            input_dim=input_dim,
            output_dim=128,
            mean_init=0.0,
            std_init=0.1,
            drive=drive
        )
        self.layer_2 = LinearModel(
            input_dim=128,
            output_dim=64,
            mean_init=0.0,
            std_init=0.1,
            drive=drive
        )
        self.layer_3 = LinearModel(
            input_dim=64,
            output_dim=output_dim,
            mean_init=0.0,
            std_init=0.1,
            drive=drive
        )
        self.relu = ReluActivateFunction()
        self.sigmoid = SigmoidActivateFunction()
        self.tanh = TanhActivateFunction()
        self.softmax = softmaxActivateFunction()
        self.cache = {}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out1 = self.layer_1.fit(X)
        act1 = self.relu.activate(out1)
        out2 = self.layer_2.fit(act1)
        act2 = self.sigmoid.activate(out2)
        out3 = self.layer_3.fit(act2)
        act3 = self.softmax.activate(out3)

        self.cache = {
            "inputs": X,
            "z1": out1,
            "a1": act1,
            "z2": out2,
            "a2": act2,
            "z3": out3,
            "a3": act3
        }
        return act3

    def backward(self, y_true: torch.Tensor = None, grad_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.cache["a3"]
        batch_size = probs.shape[0]
        if grad_output is None:
            grad_z3 = (probs - y_true) / batch_size
        else:
            grad_z3 = grad_output

        a2 = self.cache["a2"]
        self.layer_3.grad_weights.copy_(torch.mm(a2.t(), grad_z3))
        self.layer_3.grad_bias.copy_(grad_z3.sum(dim=0, keepdim=True))
        grad_a2 = torch.mm(grad_z3, self.layer_3.weights.t())

        z2 = self.cache["z2"]
        grad_z2 = grad_a2 * self.sigmoid.derivative(z2)
        a1 = self.cache["a1"]
        self.layer_2.grad_weights.copy_(torch.mm(a1.t(), grad_z2))
        self.layer_2.grad_bias.copy_(grad_z2.sum(dim=0, keepdim=True))
        grad_a1 = torch.mm(grad_z2, self.layer_2.weights.t())

        z1 = self.cache["z1"]
        grad_z1 = grad_a1 * self.relu.derivative(z1)
        inputs = self.cache["inputs"]
        self.layer_1.grad_weights.copy_(torch.mm(inputs.t(), grad_z1))
        self.layer_1.grad_bias.copy_(grad_z1.sum(dim=0, keepdim=True))
        grad_input = torch.mm(grad_z1, self.layer_1.weights.t())
        return grad_input

    def zero_grad(self):
        for layer in (self.layer_1, self.layer_2, self.layer_3):
            layer.zero_grad()

    def parameters(self):
        params = []
        for layer in (self.layer_1, self.layer_2, self.layer_3):
            params.extend(layer.parameters())
        return params

    @staticmethod
    def cross_entropy(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        preds = torch.clamp(preds, eps, 1.0 - eps)
        loss = -(labels * torch.log(preds)).sum(dim=1)
        return loss.mean()

    def train_step(self, X: torch.Tensor, y_true: torch.Tensor, optimizer, zero_grad: bool = True):
        if zero_grad:
            self.zero_grad()
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()
        preds = self.forward(X)
        loss = self.cross_entropy(preds, y_true)
        self.backward(y_true=y_true)
        optimizer.step()
        return loss, preds
