"""
Weights & Biases logging utilities.
"""

import wandb

from src.utils.config_loader import load_configs


def initialize_wandb() -> None:
    """
    Initialize a Weights & Biases run.
    """

    config = load_configs()

    wandb_config = config["wandb"]

    if not wandb_config["enabled"]:
        return

    wandb.init(
        project=wandb_config["project"],
        name=wandb_config["run_name"],
        tags=wandb_config["tags"],
        notes=wandb_config["notes"],
        config=config,
    )

    print("Weights & Biases initialized.")


def log_metrics(
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
) -> None:
    """
    Log training metrics.
    """

    config = load_configs()

    if not config["wandb"]["enabled"]:
        return

    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
        }
    )


def finish_wandb() -> None:
    """
    Finish the W&B run.
    """

    config = load_configs()

    if not config["wandb"]["enabled"]:
        return

    wandb.finish()

    print("Weights & Biases run finished.")