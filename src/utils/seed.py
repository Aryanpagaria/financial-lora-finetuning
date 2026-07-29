"""
Utilities for reproducible experiments.
"""

import os
import random

import numpy as np
import torch

from src.utils.config_loader import load_configs


def set_seed() -> None:
    """
    Set random seeds for reproducible experiments.
    """

    config = load_configs()

    seed = config["training"]["random_seed"]

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_seed_info() -> None:
    """
    Print the configured random seed.
    """

    config = load_configs()

    print("=" * 80)
    print("RANDOM SEED")
    print("=" * 80)
    print(f"Seed : {config['training']['random_seed']}")
    print("=" * 80)


if __name__ == "__main__":
    set_seed()
    print_seed_info()