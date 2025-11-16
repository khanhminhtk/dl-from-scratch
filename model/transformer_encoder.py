from typing import Optional

import torch

from model.multi_head_attention import MultiHeadAttention
from model.lazylinear import LinearModel
from model.activate_function import ReluActivateFunction
from model.layernorm import AddNorm

class FeedForwardNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, drive: str, mean_init: float = 0.0, std_init: float = 0.1) -> None:
        self.drive = drive
        self.linear1 = LinearModel(
            input_dim=input_dim,
            output_dim=hidden_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.linear2 = LinearModel(
            input_dim=hidden_dim,
            output_dim=input_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.relu = ReluActivateFunction()
        self.cache = {}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out1 = self.linear1.fit(X)
        act1 = self.relu.activate(out1)
        out2 = self.linear2.fit(act1)

        self.cache = {
            "inputs": X,
            "z1": out1,
            "a1": act1,
            "z2": out2
        }
        return out2

class TransformerEncoder:
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dim: int,
        drive: str,
        mean_init: float = 0,
        std_init: float = 0.1
    ) -> None:
        self.drive = drive
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            input_dim=input_dim,
            drive=drive,
            mean=mean_init,
            std_init=std_init
        )
        self.add_norm1 = AddNorm(dim=input_dim, drive=drive)
        self.ffn = FeedForwardNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drive=drive,
            mean_init=mean_init,
            std_init=std_init
        )
        self.add_norm2 = AddNorm(dim=input_dim, drive=drive)
        self.cache = {}

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attention.forward(X, attention_mask=attention_mask)
        out1 = self.add_norm1.forward(X, attn_output)
        ffn_output = self.ffn.forward(out1)
        out2 = self.add_norm2.forward(out1, ffn_output)

        self.cache = {
            "inputs": X,
            "attn_output": attn_output,
            "out1": out1,
            "ffn_output": ffn_output,
            "out2": out2
        }
        return out2