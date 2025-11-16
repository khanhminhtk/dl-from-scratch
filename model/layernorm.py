import torch

class LayerNorm:
    def __init__(self, dim, eps=1e-5, drive="cpu"):
        self.gamma = torch.ones(1, 1, dim, device=drive)
        self.beta = torch.zeros(1, 1, dim, device=drive)
        self.eps = eps
        self.grad_gamma = torch.zeros_like(self.gamma)
        self.grad_beta = torch.zeros_like(self.beta)

    def forward(self, X):
        # X: (B, L, D)
        mean = X.mean(dim=-1, keepdim=True)          # (B, L, 1)
        var = X.var(dim=-1, keepdim=True, unbiased=False)  # (B, L, 1)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)    # (B, L, D)
        out = self.gamma * X_hat + self.beta
        return out
    
class AddNorm:
    def __init__(self, dim, eps=1e-5, drive="cpu"):
        self.layer_norm = LayerNorm(dim, eps, drive)

    def forward(self, X, sublayer_output):
        # X: (B, L, D)
        # sublayer_output: (B, L, D)
        out = X + sublayer_output
        out = self.layer_norm.forward(out)
        return out
