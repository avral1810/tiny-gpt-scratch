import torch
from torch import nn

from tiny_gpt.bigrams import BigramLM, PositionalBigramLM
from tiny_gpt.attention import SingleHeadAttentionLM, MultiHeadAttentionLM
from tiny_gpt.losses import LanguageModelingCrossEntropyLoss


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "bigram": BigramLM,
    "positional_bigram": PositionalBigramLM,
    "single_head_attention": SingleHeadAttentionLM,
    "multi_head_attention": MultiHeadAttentionLM
}

OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

CRITERION_REGISTRY: dict[str, type[nn.Module]] = {
    "lm_cross_entropy": LanguageModelingCrossEntropyLoss,
    "language_modeling_cross_entropy": LanguageModelingCrossEntropyLoss,
}

SCHEDULER_REGISTRY = {
    "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
}


def get_model_class(model_name: str) -> type[nn.Module]:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]


def get_optimizer_class(optimizer_name: str) -> type[torch.optim.Optimizer]:
    if optimizer_name not in OPTIMIZER_REGISTRY:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. Available optimizers: {available}"
        )
    return OPTIMIZER_REGISTRY[optimizer_name]


def get_criterion_class(criterion_name: str) -> type[nn.Module]:
    if criterion_name not in CRITERION_REGISTRY:
        available = ", ".join(sorted(CRITERION_REGISTRY))
        raise ValueError(
            f"Unknown criterion '{criterion_name}'. Available criteria: {available}"
        )
    return CRITERION_REGISTRY[criterion_name]


def get_scheduler_class(scheduler_name: str):
    if scheduler_name not in SCHEDULER_REGISTRY:
        available = ", ".join(sorted(SCHEDULER_REGISTRY))
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. Available schedulers: {available}"
        )
    return SCHEDULER_REGISTRY[scheduler_name]
