import torch

from tiny_gpt.data import get_batch


def train_one_step(
    model: torch.nn.Module,
    data: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> float:
    model.train()
    x, y = get_batch(data, batch_size=batch_size, block_size=block_size)
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
