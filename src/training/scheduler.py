from typing import Any

import torch

from src.utils.config_loader import load_configs


Scheduler = torch.optim.lr_scheduler.LambdaLR


def _get_scheduler_config() -> dict[str, Any]:
    """
    Load and validate scheduler configuration.

    Scheduler hyperparameters are stored in the training
    configuration so experiments remain reproducible.
    """

    config = load_configs()

    if "training" not in config:
        raise RuntimeError(
            "Training configuration is missing."
        )

    training_config = config["training"]

    required_keys = {
        "epochs",
        "warmup_ratio",
    }

    missing_keys = [
        key
        for key in required_keys
        if key not in training_config
    ]

    if missing_keys:
        raise RuntimeError(
            "Missing scheduler configuration keys: "
            f"{missing_keys}"
        )

    epochs = int(
        training_config["epochs"]
    )

    warmup_ratio = float(
        training_config["warmup_ratio"]
    )

    if epochs <= 0:
        raise ValueError(
            "training.epochs must be greater than zero."
        )

    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            "training.warmup_ratio must be in "
            "the range [0.0, 1.0)."
        )

    return training_config


def _calculate_training_steps(
    dataloader_length: int,
    epochs: int,
    gradient_accumulation_steps: int,
) -> int:
    """
    Calculate the total number of optimizer update
    steps for the complete training run.
    """

    if dataloader_length <= 0:
        raise ValueError(
            "Dataloader length must be greater than zero."
        )

    if epochs <= 0:
        raise ValueError(
            "Epoch count must be greater than zero."
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient accumulation steps must be "
            "greater than zero."
        )

    optimizer_steps_per_epoch = (
        dataloader_length
        + gradient_accumulation_steps
        - 1
    ) // gradient_accumulation_steps

    total_training_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    if total_training_steps <= 0:
        raise RuntimeError(
            "Calculated training steps are invalid."
        )

    return total_training_steps


def _calculate_warmup_steps(
    total_training_steps: int,
    warmup_ratio: float,
) -> int:
    """
    Convert the configured warmup ratio into an
    integer number of optimizer warmup steps.
    """

    if total_training_steps <= 0:
        raise ValueError(
            "Total training steps must be greater than zero."
        )

    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            "Warmup ratio must be in "
            "the range [0.0, 1.0)."
        )

    warmup_steps = int(
        total_training_steps
        * warmup_ratio
    )

    if (
        warmup_ratio > 0.0
        and warmup_steps == 0
    ):
        warmup_steps = 1

    if warmup_steps >= total_training_steps:
        warmup_steps = (
            total_training_steps - 1
        )

    return warmup_steps


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_training_steps: int,
) -> Scheduler:
    """
    Create a linear warmup followed by linear decay
    learning-rate scheduler.

    The learning rate starts at a small fraction of
    the configured learning rate, increases during
    warmup, and then decreases linearly toward zero.
    """

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "optimizer must be a torch optimizer."
        )

    if total_training_steps <= 0:
        raise ValueError(
            "Total training steps must be greater than zero."
        )

    if warmup_steps < 0:
        raise ValueError(
            "Warmup steps cannot be negative."
        )

    if warmup_steps >= total_training_steps:
        raise ValueError(
            "Warmup steps must be smaller than "
            "total training steps."
        )

    def learning_rate_lambda(
        current_step: int,
    ) -> float:
        """
        Calculate the multiplicative learning-rate
        factor for the current optimizer step.
        """

        if current_step < warmup_steps:

            if warmup_steps == 0:
                return 1.0

            return float(
                current_step + 1
            ) / float(
                warmup_steps
            )

        decay_steps = (
            total_training_steps
            - warmup_steps
        )

        if decay_steps <= 0:
            return 0.0

        remaining_steps = (
            total_training_steps
            - current_step
        )

        return max(
            0.0,
            float(
                remaining_steps
            )
            / float(
                decay_steps
            ),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_lambda,
    )

    return scheduler


def _validate_scheduler(
    scheduler: Scheduler,
    total_training_steps: int,
    warmup_steps: int,
) -> None:
    """
    Validate the constructed scheduler.
    """

    if not isinstance(
        scheduler,
        torch.optim.lr_scheduler.LambdaLR,
    ):
        raise RuntimeError(
            "Scheduler is not a LambdaLR instance."
        )

    if total_training_steps <= 0:
        raise ValueError(
            "Total training steps must be greater than zero."
        )

    if warmup_steps < 0:
        raise ValueError(
            "Warmup steps cannot be negative."
        )

    if warmup_steps >= total_training_steps:
        raise ValueError(
            "Warmup steps must be smaller than "
            "total training steps."
        )

    if not scheduler.optimizer.param_groups:
        raise RuntimeError(
            "Scheduler optimizer contains no "
            "parameter groups."
        )

    for index, parameter_group in enumerate(
        scheduler.optimizer.param_groups
    ):

        learning_rate = parameter_group.get(
            "lr"
        )

        if learning_rate is None:
            raise RuntimeError(
                f"Parameter group {index} has "
                "no learning rate."
            )

        if learning_rate < 0:
            raise RuntimeError(
                f"Parameter group {index} has "
                "an invalid learning rate."
            )

    print("=" * 80)
    print("SCHEDULER VALIDATION")
    print("=" * 80)

    print(
        "Scheduler Type    : LambdaLR"
    )

    print(
        f"Total Steps       : "
        f"{total_training_steps:,}"
    )

    print(
        f"Warmup Steps      : "
        f"{warmup_steps:,}"
    )

    print(
        "Warmup Strategy   : Linear"
    )

    print(
        "Decay Strategy    : Linear"
    )

    print(
        "Parameter Groups  : PASSED"
    )

    print(
        "Status            : VALID"
    )

    print("=" * 80)


def _print_scheduler_configuration(
    scheduler: Scheduler,
    total_training_steps: int,
    warmup_steps: int,
) -> None:
    """
    Display the effective scheduler configuration.
    """

    if not scheduler.optimizer.param_groups:
        raise RuntimeError(
            "Cannot inspect scheduler configuration "
            "without optimizer parameter groups."
        )

    current_learning_rates = (
        scheduler.get_last_lr()
    )

    print("=" * 80)
    print("SCHEDULER CONFIGURATION")
    print("=" * 80)

    print(
        "Scheduler        : "
        f"{scheduler.__class__.__name__}"
    )

    print(
        f"Total Steps     : "
        f"{total_training_steps:,}"
    )

    print(
        f"Warmup Steps    : "
        f"{warmup_steps:,}"
    )

    print(
        "Warmup          : Linear"
    )

    print(
        "Decay           : Linear"
    )

    for index, learning_rate in enumerate(
        current_learning_rates
    ):

        print(
            f"Group {index + 1} "
            f"Initial LR : {learning_rate:.8f}"
        )

    print("=" * 80)


def sanity_check_scheduler(
    scheduler: Scheduler,
) -> None:
    """
    Perform final scheduler sanity checks.
    """

    if not isinstance(
        scheduler,
        torch.optim.lr_scheduler.LambdaLR,
    ):
        raise RuntimeError(
            "Invalid scheduler object."
        )

    if not scheduler.optimizer.param_groups:
        raise RuntimeError(
            "Scheduler optimizer contains no "
            "parameter groups."
        )

    learning_rates = scheduler.get_last_lr()

    if not learning_rates:
        raise RuntimeError(
            "Scheduler returned no learning rates."
        )

    for learning_rate in learning_rates:

        if learning_rate < 0:
            raise RuntimeError(
                "Scheduler produced a negative "
                "learning rate."
            )

    print("=" * 80)
    print("SCHEDULER SANITY CHECK")
    print("=" * 80)

    print(
        "Scheduler Object : PASSED"
    )

    print(
        "Learning Rates   : PASSED"
    )

    print(
        "Parameter Groups : PASSED"
    )

    print(
        "Status           : READY FOR TRAINING"
    )

    print("=" * 80)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    dataloader_length: int,
    gradient_accumulation_steps: int,
) -> Scheduler:
    """
    Build, validate, and return the complete learning-rate
    scheduler for the training run.
    """

    config = _get_scheduler_config()

    epochs = int(
        config["epochs"]
    )

    warmup_ratio = float(
        config["warmup_ratio"]
    )

    total_training_steps = (
        _calculate_training_steps(
            dataloader_length=dataloader_length,
            epochs=epochs,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
        )
    )

    warmup_steps = (
        _calculate_warmup_steps(
            total_training_steps=(
                total_training_steps
            ),
            warmup_ratio=warmup_ratio,
        )
    )

    scheduler = _create_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_training_steps=(
            total_training_steps
        ),
    )

    _validate_scheduler(
        scheduler=scheduler,
        total_training_steps=(
            total_training_steps
        ),
        warmup_steps=warmup_steps,
    )

    _print_scheduler_configuration(
        scheduler=scheduler,
        total_training_steps=(
            total_training_steps
        ),
        warmup_steps=warmup_steps,
    )

    sanity_check_scheduler(
        scheduler
    )

    return scheduler


def _test_scheduler() -> None:
    """
    Run an isolated scheduler test without loading
    the Qwen model or requiring a GPU.

    This validates scheduler mathematics and API
    behavior before model integration.
    """

    test_parameter = torch.nn.Parameter(
        torch.tensor(
            1.0,
            dtype=torch.float32,
        )
    )

    optimizer = torch.optim.AdamW(
        [test_parameter],
        lr=2e-4,
        weight_decay=0.01,
    )

    dataloader_length = 100

    epochs = 3

    gradient_accumulation_steps = 4

    total_training_steps = (
        _calculate_training_steps(
            dataloader_length=dataloader_length,
            epochs=epochs,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
        )
    )

    warmup_steps = (
        _calculate_warmup_steps(
            total_training_steps=(
                total_training_steps
            ),
            warmup_ratio=0.10,
        )
    )

    scheduler = _create_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_training_steps=(
            total_training_steps
        ),
    )

    initial_learning_rate = (
        scheduler.get_last_lr()[0]
    )

    if initial_learning_rate <= 0:
        raise RuntimeError(
            "Initial scheduler learning rate "
            "is invalid."
        )

    observed_learning_rates = [
        initial_learning_rate
    ]

    for _ in range(
        total_training_steps
    ):

        optimizer.step()

        scheduler.step()

        observed_learning_rates.append(
            scheduler.get_last_lr()[0]
        )

    final_learning_rate = (
        scheduler.get_last_lr()[0]
    )

    if final_learning_rate != 0.0:
        raise RuntimeError(
            "Scheduler did not decay the learning "
            "rate to zero."
        )

    if max(
        observed_learning_rates
    ) <= initial_learning_rate:
        raise RuntimeError(
            "Scheduler warmup did not increase "
            "the learning rate."
        )

    print("=" * 80)
    print("SCHEDULER ISOLATED TEST")
    print("=" * 80)

    print(
        f"Dataloader Length : "
        f"{dataloader_length}"
    )

    print(
        f"Epochs            : "
        f"{epochs}"
    )

    print(
        f"Gradient Accumulation : "
        f"{gradient_accumulation_steps}"
    )

    print(
        f"Total Training Steps : "
        f"{total_training_steps}"
    )

    print(
        f"Warmup Steps         : "
        f"{warmup_steps}"
    )

    print(
        f"Initial Learning Rate : "
        f"{initial_learning_rate:.8f}"
    )

    print(
        f"Maximum Learning Rate : "
        f"{max(observed_learning_rates):.8f}"
    )

    print(
        f"Final Learning Rate   : "
        f"{final_learning_rate:.8f}"
    )

    print(
        "Warmup Behaviour      : PASSED"
    )

    print(
        "Decay Behaviour       : PASSED"
    )

    print(
        "Status                : PASSED"
    )

    print("=" * 80)


def main() -> None:
    """
    Run the scheduler module test.
    """

    print("=" * 80)
    print("SCHEDULER MODULE TEST")
    print("=" * 80)

    _test_scheduler()

    print(
        "Scheduler module is ready "
        "for optimizer integration."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()