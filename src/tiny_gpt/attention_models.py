import torch.nn as nn
import torch


from tiny_gpt.model_blocks import (
    SelfAttentionHead,
    MultiHeadAttention,
    TransformerBlock
)

class SingleHeadAttentionLM(nn.Module):
    
    def __init__(self, vocab_size: int, block_size: int, emb_size: int, head_dim: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, emb_size,
        )
        self.positional_embedding_table = nn.Embedding(
            block_size, emb_size
        )
        # basically change this block to SA, MA, Transformer blah blah
        self.attention = SelfAttentionHead(block_size, emb_size, head_dim)
        self.lm_head = nn.Linear(head_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx)
        
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.positional_embedding_table(positions)

        raw_embeddings = tok_embeddings + pos_embeddings
        attn_out = self.attention(raw_embeddings)
        return self.lm_head(attn_out)
            
class MultiHeadAttentionLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, emb_size: int, num_heads: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, emb_size
        )
        self.positional_embedding_table = nn.Embedding(
            block_size, emb_size
        )
        self.attention = MultiHeadAttention(block_size, emb_size, num_heads)
    
        self.block_size = block_size
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx)
        
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.positional_embedding_table(positions)

        raw_embeddings = tok_embeddings + pos_embeddings
        attn_out = self.attention(raw_embeddings)
        return self.lm_head(attn_out)

class TransformersLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_size: int,
        num_heads: int
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, emb_size
        )
        self.positional_embedding_table = nn.Embedding(
            block_size, emb_size
        )
        self.attention = TransformerBlock(block_size, emb_size, num_heads)
    
        self.block_size = block_size
        self.lm_head = nn.Linear(emb_size, vocab_size)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.positional_embedding_table(positions)
        raw_embeddings = tok_embeddings + pos_embeddings
        attn_out = self.attention(raw_embeddings)
        return self.lm_head(attn_out)

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, emb_size
        )
        self.positional_embedding_table = nn.Embedding(
            block_size, emb_size
        )
        self.attention_blocks = nn.Sequential(
            *[
                TransformerBlock(block_size, emb_size, num_heads)
                for _ in range(num_layers)
            ]
        )
    
        self.block_size = block_size
        self.ln = nn.LayerNorm(emb_size) 
        self.lm_head = nn.Linear(emb_size, vocab_size)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.positional_embedding_table(positions)
        raw_embeddings = tok_embeddings + pos_embeddings
        attn_out = self.attention_blocks(raw_embeddings)
        attn_out_normed = self.ln(attn_out)
        return self.lm_head(attn_out_normed)