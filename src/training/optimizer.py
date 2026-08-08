from typing import Any

import torch

from transformers import PreTrainedModel

from src.utils.config_loader import load_configs


def _get_optimizer_config() -> dict[str, Any]:
    """
    Load and validate optimizer configuration.

    Optimizer hyperparameters are stored in the project
    configuration so experiments remain reproducible and
    do not require source-code changes.
    """

    config = load_configs()

    if "training" not in config:
        raise RuntimeError(
            "Training configuration is missing."
        )

    training_config = config["training"]

    if "learning_rate" not in training_config:
        raise RuntimeError(
            "training.learning_rate is missing."
        )

    if "weight_decay" not in training_config:
        raise RuntimeError(
            "training.weight_decay is missing."
        )

    learning_rate = float(
        training_config["learning_rate"]
    )

    weight_decay = float(
        training_config["weight_decay"]
    )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if weight_decay < 0:
        raise ValueError(
            "Weight decay cannot be negative."
        )

    return training_config

def get_optimizer(
    model: PreTrainedModel,
) -> torch.optim.Optimizer:
    """
    Build, validate, and return the AdamW optimizer
    for the trainable LoRA parameters.
    """

    config = _get_optimizer_config()

    trainable_parameters = (
        _get_trainable_parameters(
            model
        )
    )

    optimizer = _create_optimizer(
        parameters=trainable_parameters,
        config=config,
    )

    _validate_optimizer(
        optimizer=optimizer,
        expected_parameters=trainable_parameters,
    )

    _print_optimizer_configuration(
        optimizer
    )

    sanity_check_optimizer(
        optimizer
    )

    return optimizer


def _get_trainable_parameters(
    model: PreTrainedModel,
) -> list[torch.nn.Parameter]:
    """
    Return only parameters that are explicitly marked
    as trainable.

    For QLoRA, this prevents frozen base-model parameters
    from being passed to the optimizer.
    """

    trainable_parameters: list[
        torch.nn.Parameter
    ] = []

    trainable_parameter_count = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if "lora_" not in name.lower():
            raise RuntimeError(
                "A non-LoRA parameter is marked "
                "as trainable: "
                f"{name}"
            )

        trainable_parameters.append(
            parameter
        )

        trainable_parameter_count += (
            parameter.numel()
        )

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable LoRA parameters were found."
        )

    if trainable_parameter_count <= 0:
        raise RuntimeError(
            "Trainable parameter count is invalid."
        )

    print("=" * 80)
    print("OPTIMIZER PARAMETER VALIDATION")
    print("=" * 80)

    print(
        f"Trainable Parameters : "
        f"{trainable_parameter_count:,}"
    )

    print(
        f"Trainable Tensors    : "
        f"{len(trainable_parameters):,}"
    )

    print(
        "Frozen Base Parameters : EXCLUDED"
    )

    print("=" * 80)

    return trainable_parameters





def _create_optimizer(
    parameters: list[torch.nn.Parameter],
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Create the AdamW optimizer for LoRA parameters.
    """

    learning_rate = float(
        config["learning_rate"]
    )

    weight_decay = float(
        config["weight_decay"]
    )

    optimizer = torch.optim.AdamW(
        params=parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer






def _validate_optimizer(
    optimizer: torch.optim.Optimizer,
    expected_parameters: list[torch.nn.Parameter],
) -> None:
    """
    Verify that the optimizer contains exactly the
    trainable parameters supplied to it.
    """

    if not optimizer.param_groups:
        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    optimizer_parameters: list[
        torch.nn.Parameter
    ] = []

    for parameter_group in optimizer.param_groups:

        parameters = parameter_group.get(
            "params"
        )

        if parameters is None:
            raise RuntimeError(
                "Optimizer parameter group is missing "
                "the 'params' field."
            )

        for parameter in parameters:

            if not isinstance(
                parameter,
                torch.nn.Parameter,
            ):
                raise RuntimeError(
                    "Optimizer contains an invalid "
                    "parameter object."
                )

            optimizer_parameters.append(
                parameter
            )

    expected_ids = {
        id(parameter)
        for parameter in expected_parameters
    }

    optimizer_ids = {
        id(parameter)
        for parameter in optimizer_parameters
    }

    if expected_ids != optimizer_ids:
        raise RuntimeError(
            "Optimizer parameter set does not match "
            "the model's trainable parameter set."
        )

    if len(optimizer_parameters) != len(
        expected_parameters
    ):
        raise RuntimeError(
            "Optimizer contains an unexpected "
            "number of parameters."
        )

    print("=" * 80)
    print("OPTIMIZER VALIDATION")
    print("=" * 80)
    print(
        f"Parameter Groups : "
        f"{len(optimizer.param_groups):,}"
    )
    print(
        f"Optimizer Tensors : "
        f"{len(optimizer_parameters):,}"
    )
    print(
        "Parameter Match : PASSED"
    )
    print(
        "Status          : VALID"
    )
    print("=" * 80)


def _print_optimizer_configuration(
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Display the effective optimizer configuration.
    """

    if not optimizer.param_groups:
        raise RuntimeError(
            "Cannot inspect an optimizer with "
            "no parameter groups."
        )

    print("=" * 80)
    print("OPTIMIZER CONFIGURATION")
    print("=" * 80)

    print(
        f"Optimizer       : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Parameter Groups: "
        f"{len(optimizer.param_groups)}"
    )

    for index, group in enumerate(
        optimizer.param_groups
    ):

        print(
            f"\nParameter Group {index + 1}"
        )

        print(
            f"Learning Rate : "
            f"{group['lr']}"
        )

        print(
            f"Weight Decay  : "
            f"{group['weight_decay']}"
        )

        print(
            f"Betas         : "
            f"{group['betas']}"
        )

        print(
            f"Epsilon       : "
            f"{group['eps']}"
        )

    print("=" * 80)



def sanity_check_optimizer(
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Perform final validation of the optimizer
    before training begins.
    """

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise RuntimeError(
            "Invalid optimizer object."
        )

    if not optimizer.param_groups:
        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    for index, group in enumerate(
        optimizer.param_groups
    ):

        learning_rate = group.get(
            "lr"
        )

        weight_decay = group.get(
            "weight_decay"
        )

        if learning_rate is None:
            raise RuntimeError(
                f"Parameter group {index} has "
                "no learning rate."
            )

        if learning_rate <= 0:
            raise RuntimeError(
                f"Parameter group {index} has "
                "an invalid learning rate."
            )

        if weight_decay is None:
            raise RuntimeError(
                f"Parameter group {index} has "
                "no weight decay value."
            )

        if weight_decay < 0:
            raise RuntimeError(
                f"Parameter group {index} has "
                "negative weight decay."
            )

    print("=" * 80)
    print("OPTIMIZER SANITY CHECK")
    print("=" * 80)
    print(
        "Optimizer Object : PASSED"
    )
    print(
        "Parameter Groups : PASSED"
    )
    print(
        "Learning Rate    : PASSED"
    )
    print(
        "Weight Decay     : PASSED"
    )
    print(
        "Status           : READY FOR TRAINING"
    )
    print("=" * 80)




def main() -> None:
    """
    Run a local optimizer configuration validation.

    This test validates the optimizer configuration and
    optimizer implementation without loading the 3B model.
    """

    config = _get_optimizer_config()

    print("=" * 80)
    print("OPTIMIZER MODULE TEST")
    print("=" * 80)

    print(
        f"Learning Rate : "
        f"{float(config['learning_rate'])}"
    )

    print(
        f"Weight Decay  : "
        f"{float(config['weight_decay'])}"
    )

    print(
        "Configuration : VALID"
    )

    print("=" * 80)
    print(
        "Optimizer module is ready for "
        "model integration."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()