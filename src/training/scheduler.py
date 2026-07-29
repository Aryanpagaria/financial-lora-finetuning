"""
Learning rate scheduler utilities.
"""

from torch.optim import Optimizer
from transformers import get_scheduler

from src.utils.config_loader import load_configs


def get_lr_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
):
    """
    Create and return the learning rate scheduler.
    """

    config = load_configs()

    scheduler_name = (
        config["scheduler"]["name"]
        .strip()
        .lower()
    )

    warmup_ratio = config["scheduler"]["warmup_ratio"]

    warmup_steps = int(
        warmup_ratio * num_training_steps
    )

    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler


def print_scheduler_info(
    num_training_steps: int,
) -> None:
    """
    Print scheduler configuration.
    """

    config = load_configs()

    warmup_ratio = config["scheduler"]["warmup_ratio"]

    warmup_steps = int(
        warmup_ratio * num_training_steps
    )

    print("=" * 80)
    print("SCHEDULER CONFIGURATION")
    print("=" * 80)

    print(f"Scheduler      : {config['scheduler']['name']}")
    print(f"Warmup Ratio   : {warmup_ratio}")
    print(f"Warmup Steps   : {warmup_steps}")
    print(f"Training Steps : {num_training_steps}")

    print("=" * 80)


if __name__ == "__main__":

    print_scheduler_info(1000)