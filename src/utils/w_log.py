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

        print("W&B is disabled.")

        return

    if wandb.run is not None:

        print("W&B run already exists.")

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
    gradient_norm: float,
    global_step: int,
) -> None:
    """
    Log training metrics to W&B.
    """

    config = load_configs()

    if not config["wandb"]["enabled"]:

        return

    if wandb.run is None:

        return

    wandb.log(
        {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }
    )

def finish_wandb() -> None:
    """
    Finish the current W&B run.
    """

    config = load_configs()

    if not config["wandb"]["enabled"]:

        return

    if wandb.run is None:

        return

    wandb.finish()

    print("Weights & Biases run finished.")