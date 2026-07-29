"""
Device utilities for model training and inference.
"""

from typing import Final

import torch


CPU: Final[str] = "cpu"
CUDA: Final[str] = "cuda"
MPS: Final[str] = "mps"


def get_device() -> torch.device:
    """
    Return the best available device.

    Priority:
        CUDA (NVIDIA GPU)
        MPS (Apple Silicon)
        CPU
    """

    if torch.cuda.is_available():
        return torch.device(CUDA)

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device(MPS)

    return torch.device(CPU)


def get_device_name() -> str:
    """
    Return a human-readable device name.
    """

    device = get_device()

    if device.type == CUDA:
        return torch.cuda.get_device_name(0)

    if device.type == MPS:
        return "Apple Silicon GPU"

    return "CPU"


def print_device_info() -> None:
    """
    Print information about the selected device.
    """

    device = get_device()

    print("=" * 80)
    print("DEVICE INFORMATION")
    print("=" * 80)

    print(f"Device      : {device}")
    print(f"Device Name : {get_device_name()}")

    if device.type == CUDA:
        memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(f"GPU Memory  : {memory:.2f} GB")

    print("=" * 80)


if __name__ == "__main__":
    print_device_info()