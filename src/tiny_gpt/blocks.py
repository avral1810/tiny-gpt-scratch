import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, context_window, emb_size, head_dim):
        super().__init__()
        self.keys = nn.Linear(emb_size, head_dim, bias=False)
        self.query = nn.Linear(emb_size, head_dim, bias=False)
        self.values = nn.Linear(emb_size, head_dim, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_window, context_window))
        )
        self.head_dim = head_dim
        self.context_window = context_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k, q, v = self.keys(x), self.query(x), self.values(x)
        score = q @ k.transpose(-2, -1)
        score = score * self.head_dim ** -0.5
        score = score.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        score = F.softmax(score, dim=-1)
        return score @ v
