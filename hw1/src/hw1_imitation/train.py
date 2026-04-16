"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.evaluation import Logger, evaluate_policy
from hw1_imitation.model import PolicyType, build_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    print(states.shape, actions.shape, episode_ends.shape)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    # Each loader iteration yields one training batch:
    #   state: (batch_size, state_dim)
    #   action_chunk: (batch_size, chunk_size, action_dim)
    #
    # In this code, one "training step" means one batch / one optimizer update.
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    # Logger writes two things:
    # 1) a local CSV file at logger.csv_path for grading / plotting
    # 2) WandB records through logger.log(...)
    #
    # The provided Logger fixes its CSV columns from the first row it sees.
    # Since training rows and evaluation rows have different keys, we create
    # the full CSV header up front so both kinds of metrics appear:
    # - training rows fill train/* columns
    # - eval rows fill eval/mean_reward
    # - unused columns in a given row are left blank
    logger.header = [
        "train/loss",
        "train/steps_per_sec",
        "train/epoch",
        "eval/mean_reward",
        "step",
    ]

    # Pre-create the CSV with a stable column order so later logger.log(...)
    # calls can append either a training row or an evaluation row.
    with logger.csv_path.open("w") as f:
        f.write(",".join(logger.header) + "\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    def train_step(state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # Standard PyTorch update:
        # 1) clear old gradients
        # 2) compute batch loss
        # 3) backpropagate
        # 4) apply one optimizer step
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(state, action_chunk)
        loss.backward()
        optimizer.step()

        # Return a detached tensor because the caller only needs the loss value
        # for logging, not a tensor that still keeps autograd history alive.
        return loss.detach()

    compiled_train_step = train_step
    use_compiled_train_step = False
    if hasattr(torch, "compile"):
        try:
            compiled_train_step = torch.compile(train_step)
            use_compiled_train_step = True
        except Exception as exc:
            print(f"torch.compile unavailable, falling back to eager train step: {exc}")

    # global_step counts total optimizer updates across all epochs.
    global_step = 0

    # running_loss accumulates batch losses between logging events.
    running_loss = 0.0

    # interval_steps counts how many batches were processed since the last log.
    # This is usually equal to log_interval, except possibly for the last partial interval.
    interval_steps = 0

    # perf_counter() is meant for measuring elapsed durations.
    interval_start_time = time.perf_counter()
    last_eval_step: int | None = None

    # Put the model in training mode. This matters for layers such as Dropout
    # and BatchNorm. Our MSE MLP only uses Linear/ReLU, but this is still the
    # standard and correct mode before optimization.
    model.train()
    for epoch in range(config.num_epochs):
        for state, action_chunk in loader:
            # Move the current batch to the training device.
            #
            # non_blocking=True is a performance hint: if the source tensors are
            # in pinned CPU memory and the destination is CUDA, PyTorch may be
            # able to enqueue the copy asynchronously. It does not change the
            # numerical result, and on CPU it has little effect.
            state = state.to(device, non_blocking=True)
            action_chunk = action_chunk.to(device, non_blocking=True)

            try:
                loss = compiled_train_step(state, action_chunk)
            except Exception as exc:
                if not use_compiled_train_step:
                    raise
                print(
                    "Compiled train step failed at runtime; "
                    f"falling back to eager execution: {exc}"
                )
                compiled_train_step = train_step
                use_compiled_train_step = False
                loss = compiled_train_step(state, action_chunk)
            loss_value = float(loss.item())

            # One loop iteration = one batch = one training step.
            global_step += 1
            running_loss += loss_value
            interval_steps += 1

            if global_step % config.log_interval == 0:
                elapsed = time.perf_counter() - interval_start_time
                logger.log(
                    {
                        # Average batch loss over the most recent logging interval.
                        "train/loss": running_loss / interval_steps,

                        # Training throughput measured in batches per second
                        # over the most recent interval.
                        #
                        # max(elapsed, 1e-8) is just a small guard against an
                        # accidental divide-by-zero.
                        "train/steps_per_sec": interval_steps / max(elapsed, 1e-8),
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )
                running_loss = 0.0
                interval_steps = 0
                interval_start_time = time.perf_counter()

            if global_step % config.eval_interval == 0:
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger,
                )
                last_eval_step = global_step

                # evaluate_policy(...) switches the model to eval mode, so we
                # switch back before continuing training.
                model.train()

    if interval_steps > 0:
        elapsed = time.perf_counter() - interval_start_time
        logger.log(
            {
                # Log the final partial interval if training ended before the
                # next regular logging boundary.
                "train/loss": running_loss / interval_steps,
                "train/steps_per_sec": interval_steps / max(elapsed, 1e-8),
                "train/epoch": config.num_epochs,
            },
            step=global_step,
        )

    # Guarantee at least one final evaluation at the end of training, even if
    # the last step did not land exactly on eval_interval.
    if last_eval_step != global_step:
        evaluate_policy(
            model=model,
            normalizer=normalizer,
            device=device,
            chunk_size=config.chunk_size,
            video_size=config.video_size,
            num_video_episodes=config.num_video_episodes,
            flow_num_steps=config.flow_num_steps,
            step=global_step,
            logger=logger,
        )

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
