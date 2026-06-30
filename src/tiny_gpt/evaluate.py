import torch

from tiny_gpt.data import get_batch


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    criterion: torch.nn.Module,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    losses = {}
    was_training = model.training
    model.eval()

    for split, data in (("train", train_data), ("val", val_data)):
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size=batch_size, block_size=block_size)
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters

    if was_training:
        model.train()
    return losses
