import torch
import torch.nn.functional as F

from tiny_gpt.tokenizer import CharTokenizer


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)

    if was_training:
        model.train()

    return idx


def generate_samples(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    block_size: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[str]:
    samples = []
    for prompt in prompts:
        tokens = torch.tensor(
            tokenizer.encode(prompt.lower()),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        generated = generate(
            model=model,
            idx=tokens,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
        )
        samples.append(tokenizer.decode(generated[0].detach().cpu().tolist()))
    return samples
