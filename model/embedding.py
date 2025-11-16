import math
from typing import Optional

import torch
import torch.nn.functional as F


class EmbeddingLookup:
    def __init__(self, vocab_size: int, embed_dim: int, mean_init: float = 0.0, std_init: float = 0.02, drive: str = "cpu") -> None:
        self.drive = drive
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = torch.normal(
            mean=mean_init,
            std=std_init,
            size=(vocab_size, embed_dim),
            device=drive
        )
        self.grad_weight = torch.zeros_like(self.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.long()
        return F.embedding(indices, self.weight)

    def parameters(self):
        return [{"param": self.weight, "grad": self.grad_weight}]

    def zero_grad(self):
        self.grad_weight.zero_()


class PositionalEncoding:
    def __init__(self, embed_dim: int, max_len: int = 512, drive: str = "cpu") -> None:
        position = torch.arange(0, max_len, device=drive).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=drive) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim, device=drive)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.drive = drive

    def forward(self, seq_length: int) -> torch.Tensor:
        if seq_length > self.pe.size(1):
            raise ValueError("Sequence length exceeds maximum positional encoding length")
        return self.pe[:, :seq_length, :]


class BERTEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        drive: str,
        max_position_embeddings: int = 512,
        segment_vocab_size: int = 2,
        mean_init: float = 0.0,
        std_init: float = 0.02
    ) -> None:
        self.drive = drive
        self.token_embedding = EmbeddingLookup(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.segment_embedding = EmbeddingLookup(
            vocab_size=segment_vocab_size,
            embed_dim=embed_dim,
            mean_init=mean_init,
            std_init=std_init,
            drive=drive
        )
        self.position_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_position_embeddings,
            drive=drive
        )

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        token_embeddings = self.token_embedding.forward(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embeddings = self.segment_embedding.forward(token_type_ids)

        position_embeddings = self.position_encoding.forward(seq_length).to(self.drive)
        position_embeddings = position_embeddings.expand(batch_size, -1, -1)

        return token_embeddings + segment_embeddings + position_embeddings

    def parameters(self):
        params = []
        params.extend(self.token_embedding.parameters())
        params.extend(self.segment_embedding.parameters())
        return params

    def zero_grad(self):
        self.token_embedding.zero_grad()
        self.segment_embedding.zero_grad()
