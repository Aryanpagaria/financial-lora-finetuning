"""
Optimizer utilities.
"""

from typing import Any

import torch
from torch.optim import Optimizer

from src.utils.config_loader import load_configs


def get_optimizer(
    model: Any,
) -> Optimizer:
    """
    Create and return the optimizer.
    """

    config = load_configs()

    optimizer_name = (
        config["optimizer"]["name"]
        .strip()
        .lower()
    )

    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            params=trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            params=trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":

        return torch.optim.SGD(
            params=trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


def print_optimizer_info() -> None:
    """
    Print optimizer configuration.
    """

    config = load_configs()

    print("=" * 80)
    print("OPTIMIZER CONFIGURATION")
    print("=" * 80)

    print(f"Optimizer     : {config['optimizer']['name']}")
    print(f"Learning Rate : {config['training']['learning_rate']}")
    print(f"Weight Decay  : {config['training']['weight_decay']}")

    print("=" * 80)


if __name__ == "__main__":

    print_optimizer_info()