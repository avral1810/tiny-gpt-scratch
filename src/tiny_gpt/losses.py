import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelingCrossEntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, block_size, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * block_size, vocab_size)
        targets_flat = targets.reshape(batch_size * block_size)
        return F.cross_entropy(logits_flat, targets_flat)
