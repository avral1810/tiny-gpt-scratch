import torch

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    indices = torch.randint(0, data.size(0) - block_size, size=(batch_size,))
    for idx in indices:
        xs.append(data[idx: idx + block_size])
        ys.append(data[idx + 1: idx + 1 + block_size])
    return torch.stack(xs), torch.stack(ys)
