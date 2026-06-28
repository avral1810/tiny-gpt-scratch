from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
    val_loss: float,
    path: str | Path,
) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    path: str | Path,
) -> dict[str, object]:
    if isinstance(path, str):
        path = Path(path)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Loading checkpoint from {path} "
        f"w/ val_loss: {checkpoint['val_loss']} @ step: {checkpoint['step']}\n"
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "step": checkpoint["step"],
        "val_loss": checkpoint["val_loss"],
        "train_loss": checkpoint["train_loss"],
    }
