from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from tiny_gpt import (
    CharTokenizer,
    get_device,
    get_criterion_class,
    get_model_class,
    get_optimizer_class,
    get_scheduler_class,
    get_gpu_stats,
    save_checkpoint,
    set_seed,
    time_execution,
    train_one_step,
    train_val_split,
)
from tiny_gpt.evaluate import estimate_loss
from tiny_gpt.generation import generate_samples


def get_config(file_name: str = "configs/positional_bigram.yaml") -> dict:
    with open(file_name, "r", encoding="utf-8") as ip_file:
        return yaml.safe_load(ip_file)


def create_writer(config: dict) -> SummaryWriter:
    run_dir = Path(config["outputs"].get("run_dir", "runs"))
    experiment_name = config["experiment"].get("name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return SummaryWriter(run_dir / f"{experiment_name}-{timestamp}")


def read_text(path: str | Path, lowercase: bool = True) -> str:
    with open(path, "r", encoding="utf-8") as input_file:
        text = input_file.read()
    return text.lower() if lowercase else text


def prepare_data(
    input_path: str | Path,
    lowercase: bool,
    val_frac: float,
) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    raw_text = read_text(input_path, lowercase=lowercase)
    tokenizer = CharTokenizer(raw_text)
    encoded = tokenizer.encode(raw_text)
    data = torch.tensor(encoded, dtype=torch.long)
    train_data, val_data = train_val_split(data, val_frac=val_frac)
    return tokenizer, train_data, val_data


def run_training(
    config: dict,
    tokenizer: CharTokenizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
) -> None:
    data_config = config["data"]
    model_config = config["model"]
    model = get_model_class(model_config["name"])(
        vocab_size=tokenizer.vocab_size,
        **model_config.get("kwargs", {}),
    ).to(device)

    train_config = config["train"]
    criterion = get_criterion_class(train_config["criterion"])()
    optimizer = get_optimizer_class(train_config["optimizer"])(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 0.0),
    )
    scheduler = None
    scheduler_config = train_config.get("scheduler")
    if scheduler_config is not None:
        scheduler_kwargs = {
            key: value
            for key, value in scheduler_config.items()
            if key != "name"
        }
        scheduler = get_scheduler_class(scheduler_config["name"])(optimizer, **scheduler_kwargs)

    writer = create_writer(config)
    writer.add_text("config/yaml", f"```yaml\n{yaml.safe_dump(config)}\n```")

    exp_name = config["experiment"].get("name", "experiment")
    checkpoint_path = (
        Path(config["outputs"].get("checkpoint_dir", "checkpoints"))
        / f"{exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"
    )
    best_val_loss = float("inf")

    batch_size = data_config["batch_size"]
    block_size = data_config["block_size"]
    max_steps = train_config["max_steps"]
    eval_interval = train_config["eval_interval"]
    eval_iters = train_config["eval_iters"]

    print(f"===Running Experiment `{exp_name}` on {device}===")
    for step in range(max_steps + 1):
        if step % eval_interval == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                criterion=criterion,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
                device=device,
            )
            print(
                f"@Step {step} | "
                f"train_loss={losses['train']:.4f} | "
                f"val_loss={losses['val']:.4f}"
            )
            writer.add_scalar("loss/train", losses["train"], step)
            writer.add_scalar("loss/val", losses["val"], step)
            writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], step)

            for name, value in get_gpu_stats(device).items():
                writer.add_scalar(name, value, step)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                    path=checkpoint_path,
                )

        if step == max_steps:
            break

        train_one_step(
            model=model,
            data=train_data,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

    generation_config = config.get("generation")
    if generation_config is not None:
        prompts = generation_config.get("prompts", [])
        generation_block_size = generation_config.get("block_size") or block_size
        if prompts:
            for sample in generate_samples(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=generation_config.get("max_new_tokens", 200),
                block_size=generation_block_size,
                device=device,
                temperature=generation_config.get("temperature", 1.0),
                top_k=generation_config.get("top_k"),
            ):
                print(sample)
                print()

    writer.close()


@time_execution
def main(config_file_name: str = "configs/positional_bigram.yaml") -> None:
    config = get_config(file_name=config_file_name)
    set_seed(config["experiment"].get("seed", 42))

    device_str = config["runtime"].get("device", "auto")
    device = get_device() if device_str == "auto" else torch.device(device_str)

    data_config = config["data"]
    tokenizer, train_data, val_data = prepare_data(
        input_path=data_config["input_path"],
        lowercase=data_config.get("lowercase", True),
        val_frac=data_config.get("val_frac", 0.1),
    )
    run_training(
        config=config,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        device=device,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="positional_bigram.yaml",
        help="Config file name under configs/ or an explicit config path.",
    )
    args = parser.parse_args()

    config_file_name = args.config
    if "/" not in config_file_name:
        config_file_name = f"configs/{config_file_name}"

    main(config_file_name=config_file_name)
