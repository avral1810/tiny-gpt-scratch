import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super(). __init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            batch_size, block_size, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * block_size, vocab_size)
            targets_flat = targets.view(batch_size * block_size)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss


        
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

class PositionalBigramLM(nn.Module):
    
    def __init__(self, vocab_size, block_size, emb_size):
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
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # [B, T, N]
        tok_embeddings = self.token_embedding_table(idx)
        batch_size, block_size = idx.shape
        # [B, T, N]
        pos = torch.arange(block_size, device=idx.device)
        pos_embeddings = self.positional_embedding_table(pos)
        # [B, T, N]
        embeddings = tok_embeddings + pos_embeddings
        # [B, T, V]
        logits = self.linear(embeddings)
        if targets is None:
            loss = None
        else:
            batch_size, block_size, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * block_size, vocab_size)
            targets_flat = targets.view(batch_size * block_size)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss  

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx