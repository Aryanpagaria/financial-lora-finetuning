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
    Move a batch to the selected device.
    """

    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def evaluate(
    model: Any,
    dataloader: Any,
    device: torch.device,
) -> dict:
    """
    Evaluate the model and return evaluation metrics.
    """

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Evaluation",
            leave=False,
        )

        for batch in progress_bar:

            batch = move_batch_to_device(
                batch=batch,
                device=device,
            )

            outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}"
                }
            )

    average_loss = (
        total_loss /
        len(dataloader)
    )

    metrics = {
        "loss": average_loss,
    }

    return metrics