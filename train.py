"""
Entry point for training the LoRA model.
"""

from src.training.trainer import train


def main() -> None:
    """Start the training pipeline."""
    train()


if __name__ == "__main__":
    main()