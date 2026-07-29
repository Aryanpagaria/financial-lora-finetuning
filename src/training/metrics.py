"""
Utilities for tracking training metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrainingMetrics:
    """
    Store and track training metrics.
    """

    train_losses: List[float] = field(default_factory=list)
    validation_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    global_steps: List[int] = field(default_factory=list)

    best_validation_loss: float = float("inf")

    def update(
        self,
        epoch: int,
        global_step: int,
        train_loss: float,
        validation_loss: float,
        learning_rate: float,
    ) -> None:
        """
        Store metrics for one epoch.
        """

        self.epochs.append(epoch)
        self.global_steps.append(global_step)
        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)
        self.learning_rates.append(learning_rate)

        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss

    def latest(self) -> Dict[str, float]:
        """
        Return the latest recorded metrics.
        """

        return {
            "epoch": self.epochs[-1],
            "global_step": self.global_steps[-1],
            "train_loss": self.train_losses[-1],
            "validation_loss": self.validation_losses[-1],
            "learning_rate": self.learning_rates[-1],
            "best_validation_loss": self.best_validation_loss,
        }

    def print_summary(self) -> None:
        """
        Print the latest metrics.
        """

        metrics = self.latest()

        print("=" * 80)
        print("TRAINING METRICS")
        print("=" * 80)

        print(f"Epoch                : {metrics['epoch']}")
        print(f"Global Step          : {metrics['global_step']}")
        print(f"Train Loss           : {metrics['train_loss']:.4f}")
        print(f"Validation Loss      : {metrics['validation_loss']:.4f}")
        print(f"Learning Rate        : {metrics['learning_rate']:.8f}")
        print(
            f"Best Validation Loss : "
            f"{metrics['best_validation_loss']:.4f}"
        )

        print("=" * 80)