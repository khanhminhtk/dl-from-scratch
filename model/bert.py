from typing import Optional

import torch

from model.embedding import BERTEmbedding
from model.transformer_encoder import TransformerEncoder


class BERTModel:
    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        drive: str,
        max_position_embeddings: int = 512,
        segment_vocab_size: int = 2,
        mean_init: float = 0.0,
        std_init: float = 0.1
    ) -> None:
        self.drive = drive
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_dim=input_dim,
            drive=drive,
            max_position_embeddings=max_position_embeddings,
            segment_vocab_size=segment_vocab_size,
            mean_init=mean_init,
            std_init=std_init
        )
        self.layers = [
            TransformerEncoder(
                input_dim=input_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                drive=drive,
                mean_init=mean_init,
                std_init=std_init
            ) for _ in range(num_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32, device=self.drive)
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()
        attention_mask = attention_mask.to(self.drive)

        input_ids = input_ids.to(self.drive)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.drive)

        hidden_states = self.embedding.forward(input_ids, token_type_ids)
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, attention_mask=attention_mask)
        return hidden_states