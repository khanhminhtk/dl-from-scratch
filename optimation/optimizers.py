import torch
from typing import Iterable, List, Dict

from model.abstractions import AOptimizer


class Optimizer(AOptimizer):
    def __init__(self, params: Iterable[Dict[str, torch.Tensor]], lr: float = 1e-3) -> None:
        self.lr = lr
        self.param_groups: List[Dict[str, torch.Tensor]] = [dict(group) for group in params]

    def zero_grad(self):
        for group in self.param_groups:
            grad = group.get("grad")
            if grad is not None:
                grad.zero_()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def step(self):
        for group in self.param_groups:
            param = group.get("param")
            grad = group.get("grad")
            if param is None or grad is None:
                continue
            param.add_(grad, alpha=-self.lr)


class SGDMomentum(Optimizer):
    def __init__(
            self, 
            params: Iterable[Dict[str, torch.Tensor]], 
            lr: float = 1e-3, 
            momentum: float = 0.9
        ) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        for group in self.param_groups:
            param = group.get("param")
            if param is None:
                continue
            group["velocity"] = torch.zeros_like(param)

    def step(self):
        for group in self.param_groups:
            param = group.get("param")
            grad = group.get("grad")
            velocity = group.get("velocity")
            if param is None or grad is None or velocity is None:
                continue
            velocity.mul_(self.momentum).add_(grad)
            param.add_(velocity, alpha=-self.lr)


class Adam(Optimizer):
    def __init__(
            self, 
            params: Iterable[Dict[str, torch.Tensor]], 
            lr: float = 1e-3, 
            betas: tuple = (0.9, 0.999), 
            eps: float = 1e-8
        ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        for group in self.param_groups:
            param = group.get("param")
            if param is None:
                continue
            group["m"] = torch.zeros_like(param)
            group["v"] = torch.zeros_like(param)
            group["t"] = 0

    def step(self):
        for group in self.param_groups:
            param = group.get("param")
            grad = group.get("grad")
            m = group.get("m")
            v = group.get("v")
            t = group.get("t")
            if param is None or grad is None or m is None or v is None or t is None:
                continue
            t += 1
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            param.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
            group["t"] = t

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        for group in self.param_groups:
            param = group.get("param")
            if param is None:
                continue
        for group in self.param_groups:
            param = group.get("param")
            if param is not None:
                group["avg_sq"] = torch.zeros_like(param)

    def step(self):
        for group in self.param_groups:
            param, grad, avg_sq = group.get("param"), group.get("grad"), group.get("avg_sq")
            if param is None or grad is None or avg_sq is None:
                continue
            avg_sq.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
            denom = avg_sq.sqrt().add_(self.eps)
            param.addcdiv_(grad, denom, value=-self.lr)