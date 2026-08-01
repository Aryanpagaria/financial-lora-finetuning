"""
Model evaluation utilities.
"""

from typing import Any

import torch
from tqdm import tqdm


def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> dict:
    """
    Move every tensor in a batch to the target device.
    """

    moved_batch = {}

    for key, value in batch.items():

        if isinstance(
            value,
            torch.Tensor,
        ):

            moved_batch[key] = value.to(
                device,
                non_blocking=True,
            )

        else:

            moved_batch[key] = value

    return moved_batch

def evaluate(
    model: Any,
    dataloader: Any,
    device: torch.device,
) -> dict:
    """
    Evaluate the model.
    """

    if len(dataloader) == 0:

        raise RuntimeError(
            "Evaluation dataloader is empty."
        )

    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Evaluation",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():

        for batch in progress_bar:

            batch = move_batch_to_device(
                batch=batch,
                device=device,
            )

            outputs = model(
                **batch,
            )

            loss = outputs.loss

            total_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                }
            )

    average_loss = (
        total_loss
        / len(dataloader)
    )

    model.train()

    return {
        "loss": average_loss,
    }


def print_evaluation_summary(
    metrics: dict,
) -> None:
    """
    Display evaluation metrics.
    """

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"Validation Loss : "
        f"{metrics['loss']:.6f}"
    )
    print("=" * 80)