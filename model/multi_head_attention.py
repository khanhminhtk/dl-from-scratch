from typing import Optional

import torch

from model.self_attention import SelfAttention
from model.lazylinear import LinearModel

class MultiHeadAttention:
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        drive: str,
        mean: float = 0.0,
        std_init: float = 0.1
        ) -> None:
        self.drive = drive
        self.multi_heads = [
            SelfAttention(
                input_dim=input_dim,
                drive=drive,
                mean_init=mean,
                std_init=std_init
            ) for _ in range(num_heads)
        ]
        self.scale_linear = LinearModel(
            input_dim=input_dim * num_heads,
            output_dim=input_dim,
            mean_init=mean,
            std_init=std_init,
            drive=drive
        )

    
    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        head_outputs = [head.forward(X, attention_mask=attention_mask) for head in self.multi_heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.scale_linear.fit(concatenated) # shape: (batch_size, seq_length, input_dim)