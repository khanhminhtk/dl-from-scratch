import torch
import torch.nn.functional as F

from model.lazylinear import LinearModel
from model.abstractions import ASelfAttention

class SelfAttention(ASelfAttention):
    def __init__(
            self,
            input_dim: int,
            drive: str,
            mean_init: float = 0.0,
            std_init: float = 0.1
        ):
        self.drive = drive
        self.k = LinearModel(
            input_dim=input_dim,
            output_dim=input_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.q = LinearModel(
            input_dim=input_dim,
            output_dim=input_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.v = LinearModel(
            input_dim=input_dim,
            output_dim=input_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.cache = {}

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # X shape: (batch_size, seq_length, embed_dim)
        K = self.k.fit(X)
        Q = self.q.fit(X)
        V = self.v.fit(X)

        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32).to(self.drive))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 3:
                mask = attention_mask
            else:
                raise ValueError("attention_mask must have rank 2 or 3")
            mask = mask.to(device=self.drive)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        self.cache = {
            "K": K,
            "Q": Q,
            "V": V,
            "attn_weights": attn_weights
        }

        return output # shape: (batch_size, seq_length, embed_dim)
    
        