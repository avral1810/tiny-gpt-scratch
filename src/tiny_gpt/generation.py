import torch

from tiny_gpt.tokenizer import CharTokenizer


def generate_samples(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    samples = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            tokens = torch.tensor(
                tokenizer.encode(prompt.lower()),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            generated = model.generate(tokens, max_new_tokens=max_new_tokens)
            samples.append(tokenizer.decode(generated[0].detach().cpu().tolist()))
    return samples
