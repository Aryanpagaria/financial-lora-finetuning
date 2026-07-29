"""
Early stopping utilities.
"""

from src.utils.config_loader import load_configs


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    """

    def __init__(self) -> None:

        config = load_configs()

        self.enabled = config["early_stopping"]["enabled"]

        self.patience = config["early_stopping"]["patience"]

        self.min_delta = config["early_stopping"]["min_delta"]

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False

    def update(
        self,
        validation_loss: float,
    ) -> bool:
        """
        Update early stopping state.

        Returns:
            True if training should stop.
        """

        if not self.enabled:
            return False

        improvement = (
            self.best_loss -
            validation_loss
        )

        if improvement > self.min_delta:

            self.best_loss = validation_loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """
        Reset the early stopping state.
        """

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False

    def print_status(self) -> None:
        """
        Print current early stopping state.
        """

        print("=" * 80)
        print("EARLY STOPPING")
        print("=" * 80)

        print(f"Enabled      : {self.enabled}")
        print(f"Best Loss    : {self.best_loss:.4f}")
        print(f"Patience     : {self.patience}")
        print(f"Counter      : {self.counter}")
        print(f"Min Delta    : {self.min_delta}")
        print(f"Should Stop  : {self.should_stop}")

        print("=" * 80)