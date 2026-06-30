import torch
import torch.nn as nn


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_embedding_table(idx)


class PositionalBigramLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, emb_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, emb_size
        )
        self.positional_embedding_table = nn.Embedding(
            block_size, emb_size
        )
        self.linear = nn.Linear(
            emb_size, vocab_size
        )
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # [B, T, N]
        tok_embeddings = self.token_embedding_table(idx)
        _, block_size = idx.shape
        # [B, T, N]
        pos = torch.arange(block_size, device=idx.device)
        pos_embeddings = self.positional_embedding_table(pos)
        # [B, T, N]
        embeddings = tok_embeddings + pos_embeddings
        # [B, T, V]
        logits = self.linear(embeddings)
        return logits
