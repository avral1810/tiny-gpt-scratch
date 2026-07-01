import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, block_size: int, emb_size: int, head_dim: int):
        super().__init__()
        self.keys = nn.Linear(emb_size, head_dim, bias=False)
        self.query = nn.Linear(emb_size, head_dim, bias=False)
        self.values = nn.Linear(emb_size, head_dim, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )
        self.head_dim = head_dim
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k, q, v = self.keys(x), self.query(x), self.values(x)
        score = q @ k.transpose(-2, -1)
        score = score * self.head_dim ** -0.5
        score = score.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        score = F.softmax(score, dim=-1)
        return score @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int, emb_size: int, num_heads: int):
        if emb_size % num_heads != 0:
            raise ValueError("`emb_size % num_heads` should be 0")
        super().__init__()            
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(block_size, emb_size, emb_size // num_heads) 
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outs = [head(x) for head in self.heads]
        x = torch.cat(head_outs, dim=-1)
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, scale: int=4):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(emb_size, scale * emb_size),
            nn.GELU(),
            nn.Linear(scale * emb_size, emb_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffwd(x)

class TransformerBlock(nn.Module):
    def __init__(self, block_size: int, emb_size: int, num_heads: int):
        super().__init__()
        self.multi_heads = MultiHeadAttention(block_size, emb_size, num_heads)
        self.ln1 = nn.LayerNorm(emb_size) 
        self.ff = FeedForward(emb_size)
        self.ln2 = nn.LayerNorm(emb_size) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
