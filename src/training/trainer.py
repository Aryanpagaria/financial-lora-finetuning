from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.training.checkpoint import (
    find_resume_checkpoint,
    resume_from_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)
from src.utils.config_loader import load_configs
TrainingState = dict[str, Any]


def _get_training_config() -> dict[str, Any]:
    """
    Load and validate the training configuration.
    """

    config = load_configs()

    if "training" not in config:
        raise RuntimeError(
            "Training configuration is missing."
        )

    training_config = config["training"]

    if not isinstance(
        training_config,
        dict,
    ):
        raise RuntimeError(
            "Training configuration must be a dictionary."
        )

    return training_config


def _validate_training_config(
    training_config: dict[str, Any],
) -> None:
    """
    Validate the configuration required by the trainer.
    """

    required_keys = {
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
        "max_seq_length",
        "optimizer",
        "scheduler",
        "precision",
        "dataloader",
        "validation",
        "logging",
    }

    missing_keys = sorted(
        required_keys
        - set(training_config.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Missing training configuration keys: "
            f"{missing_keys}"
        )

    epochs = training_config["epochs"]

    if not isinstance(
        epochs,
        int,
    ) or epochs <= 0:
        raise ValueError(
            "training.epochs must be a positive integer."
        )

    batch_size = training_config[
        "batch_size"
    ]

    if not isinstance(
        batch_size,
        int,
    ) or batch_size <= 0:
        raise ValueError(
            "training.batch_size must be a positive integer."
        )

    gradient_accumulation_steps = (
        training_config[
            "gradient_accumulation_steps"
        ]
    )

    if (
        not isinstance(
            gradient_accumulation_steps,
            int,
        )
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError(
            "training.gradient_accumulation_steps "
            "must be a positive integer."
        )

    learning_rate = training_config[
        "learning_rate"
    ]

    if (
        not isinstance(
            learning_rate,
            (int, float),
        )
        or learning_rate <= 0
    ):
        raise ValueError(
            "training.learning_rate must be greater than zero."
        )

    weight_decay = training_config[
        "weight_decay"
    ]

    if (
        not isinstance(
            weight_decay,
            (int, float),
        )
        or weight_decay < 0
    ):
        raise ValueError(
            "training.weight_decay cannot be negative."
        )

    max_grad_norm = training_config[
        "max_grad_norm"
    ]

    if (
        not isinstance(
            max_grad_norm,
            (int, float),
        )
        or max_grad_norm <= 0
    ):
        raise ValueError(
            "training.max_grad_norm must be greater than zero."
        )

    warmup_ratio = training_config[
        "warmup_ratio"
    ]

    if (
        not isinstance(
            warmup_ratio,
            (int, float),
        )
        or not 0 <= warmup_ratio <= 1
    ):
        raise ValueError(
            "training.warmup_ratio must be between 0 and 1."
        )

    max_seq_length = training_config[
        "max_seq_length"
    ]

    if (
        not isinstance(
            max_seq_length,
            int,
        )
        or max_seq_length <= 0
    ):
        raise ValueError(
            "training.max_seq_length must be a "
            "positive integer."
        )

    precision = training_config[
        "precision"
    ]

    if not isinstance(
        precision,
        dict,
    ):
        raise ValueError(
            "training.precision must be a dictionary."
        )

    if not isinstance(
        precision.get("fp16"),
        bool,
    ):
        raise ValueError(
            "training.precision.fp16 must be boolean."
        )

    if not isinstance(
        precision.get("bf16"),
        bool,
    ):
        raise ValueError(
            "training.precision.bf16 must be boolean."
        )

    if (
        precision["fp16"]
        and precision["bf16"]
    ):
        raise ValueError(
            "FP16 and BF16 cannot both be enabled."
        )

    dataloader_config = training_config[
        "dataloader"
    ]

    if not isinstance(
        dataloader_config,
        dict,
    ):
        raise ValueError(
            "training.dataloader must be a dictionary."
        )

    num_workers = dataloader_config.get(
        "num_workers"
    )

    if (
        not isinstance(
            num_workers,
            int,
        )
        or num_workers < 0
    ):
        raise ValueError(
            "training.dataloader.num_workers must "
            "be a non-negative integer."
        )

    validation_config = training_config[
        "validation"
    ]

    if not isinstance(
        validation_config,
        dict,
    ):
        raise ValueError(
            "training.validation must be a dictionary."
        )

    logging_config = training_config[
        "logging"
    ]

    if not isinstance(
        logging_config,
        dict,
    ):
        raise ValueError(
            "training.logging must be a dictionary."
        )


def _build_dataloader(
    dataset: Dataset[Any],
    training_config: dict[str, Any],
    split_name: str,
) -> DataLoader[Any]:
    """
    Build a DataLoader for a specific dataset split.
    """

    if not isinstance(
        dataset,
        Dataset,
    ):
        raise TypeError(
            "dataset must inherit from torch.utils.data.Dataset."
        )

    if split_name not in {
        "train",
        "validation",
        "dev",
        "test",
    }:
        raise ValueError(
            "Unsupported dataset split: "
            f"{split_name}"
        )

    dataloader_config = training_config[
        "dataloader"
    ]

    if split_name == "train":

        shuffle = dataloader_config[
            "shuffle_train"
        ]

    elif split_name in {
        "validation",
        "dev",
    }:

        shuffle = dataloader_config[
            "shuffle_validation"
        ]

    else:

        shuffle = dataloader_config[
            "shuffle_test"
        ]

    num_workers = dataloader_config[
        "num_workers"
    ]

    pin_memory = dataloader_config[
        "pin_memory"
    ]

    persistent_workers = (
        dataloader_config[
            "persistent_workers"
        ]
    )

    if num_workers == 0:
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=training_config[
            "batch_size"
        ],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _create_training_state(
    start_epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
) -> TrainingState:
    """
    Create the mutable state used by the training loop.
    """

    if not isinstance(
        start_epoch,
        int,
    ) or start_epoch < 0:
        raise ValueError(
            "start_epoch must be a non-negative integer."
        )

    if not isinstance(
        global_step,
        int,
    ) or global_step < 0:
        raise ValueError(
            "global_step must be a non-negative integer."
        )

    if (
        best_metric is not None
        and not isinstance(
            best_metric,
            (int, float),
        )
    ):
        raise TypeError(
            "best_metric must be a number or None."
        )

    return {
        "epoch": start_epoch,
        "global_step": global_step,
        "best_metric": (
            None
            if best_metric is None
            else float(best_metric)
        ),
        "train_loss": None,
        "validation_loss": None,
        "learning_rate": None,
        "gradient_norm": None,
    }

def _prepare_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Validate and move a tokenized batch to the training device.
    """

    if not isinstance(
        batch,
        dict,
    ):
        raise TypeError(
            "Training batch must be a dictionary."
        )

    required_keys = {
        "input_ids",
        "attention_mask",
        "labels",
    }

    missing_keys = sorted(
        required_keys
        - set(batch.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Training batch is missing required keys: "
            f"{missing_keys}"
        )

    prepared_batch: dict[
        str,
        torch.Tensor,
    ] = {}

    for key in required_keys:

        value = batch[key]

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"Batch field '{key}' must be a torch.Tensor."
            )

        if value.ndim != 2:
            raise ValueError(
                f"Batch field '{key}' must be two-dimensional. "
                f"Received shape: {tuple(value.shape)}"
            )

        if value.numel() == 0:
            raise ValueError(
                f"Batch field '{key}' is empty."
            )

        prepared_batch[key] = value.to(
            device=device,
            non_blocking=True,
        )

    input_ids = prepared_batch[
        "input_ids"
    ]

    attention_mask = prepared_batch[
        "attention_mask"
    ]

    labels = prepared_batch[
        "labels"
    ]

    if (
        input_ids.shape
        != attention_mask.shape
    ):
        raise ValueError(
            "input_ids and attention_mask must "
            "have identical shapes."
        )

    if (
        input_ids.shape
        != labels.shape
    ):
        raise ValueError(
            "input_ids and labels must "
            "have identical shapes."
        )

    if input_ids.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            "input_ids must use an integer tensor dtype."
        )

    if attention_mask.dtype not in {
        torch.int32,
        torch.int64,
        torch.bool,
    }:
        raise TypeError(
            "attention_mask must use an integer "
            "or boolean tensor dtype."
        )

    if labels.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            "labels must use an integer tensor dtype."
        )

    return prepared_batch


def _validate_model_output_loss(
    outputs: Any,
) -> torch.Tensor:
    """
    Extract and validate the causal language-model loss.
    """

    if outputs is None:
        raise RuntimeError(
            "Model returned no output."
        )

    loss = getattr(
        outputs,
        "loss",
        None,
    )

    if loss is None:
        raise RuntimeError(
            "Model output does not contain a loss. "
            "Ensure labels are passed to the model."
        )

    if not isinstance(
        loss,
        torch.Tensor,
    ):
        raise TypeError(
            "Model loss must be a torch.Tensor."
        )

    if loss.ndim != 0:
        raise ValueError(
            "Model loss must be a scalar tensor."
        )

    if not torch.isfinite(
        loss.detach()
    ):
        raise FloatingPointError(
            "Model produced a non-finite loss."
        )

    if not loss.requires_grad:
        raise RuntimeError(
            "Model loss does not require gradients. "
            "Verify that the model is in training mode "
            "and trainable parameters are configured correctly."
        )

    return loss


def _scale_loss_for_accumulation(
    loss: torch.Tensor,
    accumulation_window_size: int,
) -> torch.Tensor:
    """
    Scale the loss by the actual number of batches contributing
    to the current gradient-accumulation window.

    This correctly handles both full accumulation windows and
    the final partial window of an epoch.
    """

    if not isinstance(
        loss,
        torch.Tensor,
    ):
        raise TypeError(
            "loss must be a torch.Tensor."
        )

    if loss.ndim != 0:
        raise ValueError(
            "loss must be a scalar tensor."
        )

    if not isinstance(
        accumulation_window_size,
        int,
    ):
        raise TypeError(
            "accumulation_window_size must be an integer."
        )

    if accumulation_window_size <= 0:
        raise ValueError(
            "accumulation_window_size must be greater than zero."
        )

    return loss / accumulation_window_size






def _should_optimizer_step(
    batch_index: int,
    total_batches: int,
    gradient_accumulation_steps: int,
) -> bool:
    """
    Determine whether the current batch should trigger
    an optimizer update.

    The final partial accumulation window also performs an
    optimizer step so gradients are not discarded.
    """

    if not isinstance(
        batch_index,
        int,
    ) or batch_index < 0:
        raise ValueError(
            "batch_index must be a non-negative integer."
        )

    if not isinstance(
        total_batches,
        int,
    ) or total_batches <= 0:
        raise ValueError(
            "total_batches must be a positive integer."
        )

    if batch_index >= total_batches:
        raise ValueError(
            "batch_index must be smaller than total_batches."
        )

    if not isinstance(
        gradient_accumulation_steps,
        int,
    ) or gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be "
            "a positive integer."
        )

    is_accumulation_boundary = (
        (batch_index + 1)
        % gradient_accumulation_steps
        == 0
    )

    is_final_batch = (
        batch_index + 1
        == total_batches
    )

    return (
        is_accumulation_boundary
        or is_final_batch
    )

def _clip_gradients(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    """
    Clip model gradients and return the resulting
    total gradient norm.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if (
        not isinstance(
            max_grad_norm,
            (int, float),
        )
        or max_grad_norm <= 0
    ):
        raise ValueError(
            "max_grad_norm must be greater than zero."
        )

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(
            max_grad_norm
        ),
    )

    gradient_norm_value = float(
        gradient_norm.detach().cpu().item()
    )

    if not torch.isfinite(
        torch.tensor(
            gradient_norm_value
        )
    ):
        raise FloatingPointError(
            "Gradient norm became non-finite."
        )

    return gradient_norm_value


def _perform_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    max_grad_norm: float,
) -> tuple[float, float]:
    """
    Clip gradients, update model parameters, advance the
    learning-rate scheduler, and clear accumulated gradients.

    Returns:
        A tuple containing:
        - gradient norm
        - current learning rate
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "optimizer must be a torch optimizer."
        )

    if not isinstance(
        scheduler,
        torch.optim.lr_scheduler.LRScheduler,
    ):
        raise TypeError(
            "scheduler must be a PyTorch learning-rate scheduler."
        )

    gradient_norm = _clip_gradients(
        model=model,
        max_grad_norm=max_grad_norm,
    )

    optimizer.step()

    scheduler.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    learning_rates = [
        float(
            parameter_group["lr"]
        )
        for parameter_group
        in optimizer.param_groups
    ]

    if not learning_rates:
        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    learning_rate = max(
        learning_rates
    )

    if not torch.isfinite(
        torch.tensor(
            learning_rate
        )
    ):
        raise FloatingPointError(
            "Optimizer produced a non-finite learning rate."
        )

    return (
        gradient_norm,
        learning_rate,
    )

def _train_single_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    accumulation_window_size: int,
) -> torch.Tensor:
    """
    Execute the forward and backward passes for one batch.

    The loss is divided by the actual number of batches in the
    current accumulation window before backpropagation.
    """

    prepared_batch = _prepare_batch(
        batch=batch,
        device=device,
    )

    model.train()

    outputs = model(
        input_ids=prepared_batch["input_ids"],
        attention_mask=prepared_batch["attention_mask"],
        labels=prepared_batch["labels"],
    )

    loss = _validate_model_output_loss(
        outputs
    )

    scaled_loss = _scale_loss_for_accumulation(
        loss=loss,
        accumulation_window_size=accumulation_window_size,
    )

    scaled_loss.backward()

    return loss.detach()



def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    global_step: int,
) -> tuple[float, int, float, float]:
    """
    Train the model for one complete epoch.

    Gradient accumulation correctly handles the final partial
    accumulation window.

    Returns:
        average_loss,
        updated_global_step,
        last_gradient_norm,
        last_learning_rate
    """

    if len(dataloader) == 0:
        raise RuntimeError(
            "Training dataloader contains no batches."
        )

    if not isinstance(
        global_step,
        int,
    ) or global_step < 0:
        raise ValueError(
            "global_step must be a non-negative integer."
        )

    if not isinstance(
        gradient_accumulation_steps,
        int,
    ) or gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be "
            "a positive integer."
        )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_batches = len(dataloader)

    last_gradient_norm = 0.0
    last_learning_rate = _get_current_learning_rate(
        optimizer
    )

    optimizer_steps = 0

    for batch_index, batch in enumerate(
        dataloader
    ):

        remaining_batches = (
            total_batches - batch_index
        )

        accumulation_window_size = min(
            gradient_accumulation_steps,
            remaining_batches,
        )

        loss = _train_single_batch(
            model=model,
            batch=batch,
            device=device,
            accumulation_window_size=(
                accumulation_window_size
            ),
        )

        loss_value = float(
            loss.detach().cpu().item()
        )

        if not torch.isfinite(
            torch.tensor(
                loss_value,
                dtype=torch.float64,
            )
        ):
            raise FloatingPointError(
                "Non-finite training loss detected."
            )

        total_loss += loss_value

        should_step = _should_optimizer_step(
            batch_index=batch_index,
            total_batches=total_batches,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
        )

        if should_step:

            (
                last_gradient_norm,
                last_learning_rate,
            ) = _perform_optimizer_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=max_grad_norm,
            )

            optimizer_steps += 1
            global_step += 1

    if optimizer_steps == 0:
        raise RuntimeError(
            "Training epoch completed without an optimizer step."
        )

    average_loss = (
        total_loss / total_batches
    )

    if not torch.isfinite(
        torch.tensor(
            average_loss,
            dtype=torch.float64,
        )
    ):
        raise FloatingPointError(
            "Average training loss is non-finite."
        )

    return (
        average_loss,
        global_step,
        last_gradient_norm,
        last_learning_rate,
    )





def _validate_single_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Execute one validation batch without gradient computation.

    Returns:
        Validation loss detached from the computation graph.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    prepared_batch = _prepare_batch(
        batch=batch,
        device=device,
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            input_ids=prepared_batch[
                "input_ids"
            ],
            attention_mask=prepared_batch[
                "attention_mask"
            ],
            labels=prepared_batch[
                "labels"
            ],
        )

    loss = getattr(
        outputs,
        "loss",
        None,
    )

    if loss is None:
        raise RuntimeError(
            "Model output does not contain a validation loss."
        )

    if not isinstance(
        loss,
        torch.Tensor,
    ):
        raise TypeError(
            "Validation loss must be a torch.Tensor."
        )

    if loss.ndim != 0:
        raise ValueError(
            "Validation loss must be a scalar tensor."
        )

    if not torch.isfinite(
        loss.detach()
    ):
        raise FloatingPointError(
            "Non-finite validation loss detected."
        )

    return loss.detach()


def _validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
) -> float:
    """
    Evaluate the model over the complete validation dataset.

    Returns:
        Mean validation loss for the epoch.
    """

    if len(dataloader) == 0:
        raise RuntimeError(
            "Validation dataloader contains no batches."
        )

    model.eval()

    total_loss = 0.0
    total_batches = len(
        dataloader
    )

    for batch in dataloader:

        loss = _validate_single_batch(
            model=model,
            batch=batch,
            device=device,
        )

        loss_value = float(
            loss.cpu().item()
        )

        if not torch.isfinite(
            torch.tensor(
                loss_value
            )
        ):
            raise FloatingPointError(
                "Non-finite validation loss detected."
            )

        total_loss += loss_value

    validation_loss = (
        total_loss
        / total_batches
    )

    if not torch.isfinite(
        torch.tensor(
            validation_loss
        )
    ):
        raise FloatingPointError(
            "Mean validation loss is non-finite."
        )

    return validation_loss


def _update_training_state(
    state: TrainingState,
    epoch: int,
    global_step: int,
    train_loss: float,
    validation_loss: float | None,
    learning_rate: float,
    gradient_norm: float,
) -> TrainingState:
    """
    Update the trainer state after an epoch.
    """

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "state must be a dictionary."
        )

    if epoch < 0:
        raise ValueError(
            "epoch cannot be negative."
        )

    if global_step < 0:
        raise ValueError(
            "global_step cannot be negative."
        )

    numeric_values = {
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "gradient_norm": gradient_norm,
    }

    if validation_loss is not None:
        numeric_values[
            "validation_loss"
        ] = validation_loss

    for name, value in numeric_values.items():

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if not torch.isfinite(
            torch.tensor(
                float(value)
            )
        ):
            raise FloatingPointError(
                f"{name} must be finite."
            )

    state["epoch"] = epoch
    state["global_step"] = global_step
    state["train_loss"] = float(
        train_loss
    )
    state["validation_loss"] = (
        None
        if validation_loss is None
        else float(validation_loss)
    )
    state["learning_rate"] = float(
        learning_rate
    )
    state["gradient_norm"] = float(
        gradient_norm
    )

    return state


def _is_best_validation_loss(
    validation_loss: float,
    best_metric: float | None,
) -> bool:
    """
    Determine whether the current validation loss is better
    than the previously recorded best validation loss.

    Lower validation loss is considered better.
    """

    if not isinstance(
        validation_loss,
        (int, float),
    ):
        raise TypeError(
            "validation_loss must be numeric."
        )

    if not torch.isfinite(
        torch.tensor(
            float(validation_loss)
        )
    ):
        raise FloatingPointError(
            "validation_loss must be finite."
        )

    if best_metric is None:
        return True

    if not isinstance(
        best_metric,
        (int, float),
    ):
        raise TypeError(
            "best_metric must be numeric or None."
        )

    if not torch.isfinite(
        torch.tensor(
            float(best_metric)
        )
    ):
        raise FloatingPointError(
            "best_metric must be finite."
        )

    return (
        validation_loss
        < best_metric
    )



def _get_current_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Return the current learning rate from the optimizer.
    """

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "optimizer must be a torch optimizer."
        )

    if not optimizer.param_groups:
        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    learning_rates = []

    for parameter_group in optimizer.param_groups:

        learning_rate = parameter_group.get(
            "lr"
        )

        if not isinstance(
            learning_rate,
            (int, float),
        ):
            raise RuntimeError(
                "Optimizer parameter group contains "
                "an invalid learning rate."
            )

        if not torch.isfinite(
            torch.tensor(
                float(learning_rate)
            )
        ):
            raise FloatingPointError(
                "Optimizer learning rate is non-finite."
            )

        learning_rates.append(
            float(learning_rate)
        )

    return max(
        learning_rates
    )


def _log_epoch_metrics(
    epoch: int,
    total_epochs: int,
    global_step: int,
    train_loss: float,
    validation_loss: float | None,
    learning_rate: float,
    gradient_norm: float,
) -> None:
    """
    Print epoch-level training metrics.

    Logging is intentionally kept independent from the
    training logic so the trainer can later be connected
    to the project's logging module.
    """

    print("=" * 80)

    print(
        f"Epoch {epoch}/{total_epochs}"
    )

    print(
        f"Global Step       : {global_step:,}"
    )

    print(
        f"Train Loss        : {train_loss:.6f}"
    )

    if validation_loss is not None:

        print(
            f"Validation Loss   : "
            f"{validation_loss:.6f}"
        )

    else:

        print(
            "Validation Loss   : N/A"
        )

    print(
        f"Learning Rate     : "
        f"{learning_rate:.10f}"
    )

    print(
        f"Gradient Norm     : "
        f"{gradient_norm:.6f}"
    )

    print("=" * 80)



def _save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    full_config: dict[str, Any],
) -> None:
    """
    Save the current training state when checkpointing is enabled.

    The complete effective project configuration is stored so that
    future resume operations can detect incompatible changes to
    model, LoRA, optimizer, scheduler, precision, and data-shape
    settings.
    """

    if not isinstance(
        full_config,
        dict,
    ):
        raise TypeError(
            "full_config must be a dictionary."
        )

    checkpoint_config = full_config.get(
        "checkpoint"
    )

    if not isinstance(
        checkpoint_config,
        dict,
    ):
        raise RuntimeError(
            "Full configuration does not contain "
            "a valid checkpoint configuration."
        )

    save_strategy = checkpoint_config.get(
        "save_strategy"
    )

    if save_strategy == "none":
        return

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=int(
            state["epoch"]
        ),
        global_step=int(
            state["global_step"]
        ),
        best_metric=state.get(
            "best_metric"
        ),
        training_config=full_config,
    )



def _process_epoch(
    model: torch.nn.Module,
    train_dataloader: DataLoader[Any],
    validation_dataloader: DataLoader[Any] | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    training_config: dict[str, Any],
    full_config: dict[str, Any],
    state: TrainingState,
) -> TrainingState:
    """
    Execute one complete training epoch, optional validation,
    best-metric tracking, logging, and checkpointing.
    """

    gradient_accumulation_steps = (
        training_config[
            "gradient_accumulation_steps"
        ]
    )

    max_grad_norm = training_config[
        "max_grad_norm"
    ]

    (
        train_loss,
        global_step,
        gradient_norm,
        learning_rate,
    ) = _train_epoch(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        max_grad_norm=max_grad_norm,
        global_step=int(
            state["global_step"]
        ),
    )

    validation_loss = None

    validation_config = training_config[
        "validation"
    ]

    validation_enabled = validation_config[
        "enabled"
    ]

    if (
        validation_enabled
        and validation_dataloader is not None
    ):

        validation_loss = _validate_epoch(
            model=model,
            dataloader=validation_dataloader,
            device=device,
        )

    state = _update_training_state(
        state=state,
        epoch=int(
            state["epoch"]
        ) + 1,
        global_step=global_step,
        train_loss=train_loss,
        validation_loss=validation_loss,
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
    )

    if validation_loss is not None:

        if _is_best_validation_loss(
            validation_loss=validation_loss,
            best_metric=state.get(
                "best_metric"
            ),
        ):

            state["best_metric"] = (
                validation_loss
            )

    _log_epoch_metrics(
        epoch=int(
            state["epoch"]
        ),
        total_epochs=int(
            training_config["epochs"]
        ),
        global_step=int(
            state["global_step"]
        ),
        train_loss=float(
            state["train_loss"]
        ),
        validation_loss=state[
            "validation_loss"
        ],
        learning_rate=float(
            state["learning_rate"]
        ),
        gradient_norm=float(
            state["gradient_norm"]
        ),
    )

    _save_epoch_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        full_config=full_config,
    )

    return state


def _resume_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    full_config: dict[str, Any],
) -> TrainingState:
    """
    Restore training state from the latest valid checkpoint.

    Resume compatibility is checked against the complete current
    project configuration before model, optimizer, scheduler, and
    RNG state are restored.
    """

    if not isinstance(
        full_config,
        dict,
    ):
        raise TypeError(
            "full_config must be a dictionary."
        )

    checkpoint_path = find_resume_checkpoint()

    if checkpoint_path is None:
        return _create_training_state()

    print("=" * 80)
    print("RESUMING TRAINING")
    print("=" * 80)

    print(
        f"Checkpoint : {checkpoint_path}"
    )

    checkpoint_state = resume_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        current_training_config=full_config,
    )

    restored_epoch = int(
        checkpoint_state["epoch"]
    )

    restored_global_step = int(
        checkpoint_state["global_step"]
    )

    restored_best_metric = checkpoint_state[
        "best_metric"
    ]

    state = _create_training_state(
        start_epoch=restored_epoch,
        global_step=restored_global_step,
        best_metric=restored_best_metric,
    )

    print(
        f"Resumed Epoch      : {restored_epoch}"
    )

    print(
        f"Resumed Global Step: {restored_global_step}"
    )

    print(
        f"Best Metric        : {restored_best_metric}"
    )

    print("=" * 80)

    return state


def _save_best_model_if_improved(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    full_config: dict[str, Any],
    previous_best_metric: float | None,
) -> None:
    """
    Save a best-model checkpoint when the current validation
    loss improves over the previous best value.
    """

    if not isinstance(
        full_config,
        dict,
    ):
        raise TypeError(
            "full_config must be a dictionary."
        )

    current_validation_loss = state.get(
        "validation_loss"
    )

    if current_validation_loss is None:
        return

    if not isinstance(
        current_validation_loss,
        (int, float),
    ):
        raise TypeError(
            "Current validation loss must be numeric."
        )

    if not torch.isfinite(
        torch.tensor(
            float(current_validation_loss)
        )
    ):
        raise FloatingPointError(
            "Current validation loss is non-finite."
        )

    if not _is_best_validation_loss(
        validation_loss=float(
            current_validation_loss
        ),
        best_metric=previous_best_metric,
    ):
        return

    best_path = save_best_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=int(
            state["epoch"]
        ),
        global_step=int(
            state["global_step"]
        ),
        best_metric=float(
            current_validation_loss
        ),
        training_config=full_config,
    )

    print(
        f"Best Checkpoint    : {best_path}"
    )


def _train_all_epochs(
    model: torch.nn.Module,
    train_dataloader: DataLoader[Any],
    validation_dataloader: DataLoader[Any] | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    training_config: dict[str, Any],
    full_config: dict[str, Any],
    state: TrainingState,
) -> TrainingState:
    """
    Execute all remaining training epochs.

    Training resumes from the epoch stored in the supplied
    state and continues until the configured epoch count.
    """

    total_epochs = int(
        training_config["epochs"]
    )

    if total_epochs <= 0:
        raise ValueError(
            "Training epochs must be greater than zero."
        )

    starting_epoch = int(
        state["epoch"]
    )

    if starting_epoch > total_epochs:
        raise RuntimeError(
            "Checkpoint epoch exceeds configured "
            f"training epochs: "
            f"{starting_epoch} > {total_epochs}"
        )

    for _ in range(
        starting_epoch,
        total_epochs,
    ):

        previous_best_metric = state.get(
            "best_metric"
        )

        state = _process_epoch(
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            training_config=training_config,
            full_config=full_config,
            state=state,
        )

        _save_best_model_if_improved(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            full_config=full_config,
            previous_best_metric=previous_best_metric,
        )

    return state


def _finalize_training(
    model: torch.nn.Module,
    state: TrainingState,
    training_config: dict[str, Any],
) -> TrainingState:
    """
    Validate and finalize the training state after the
    configured training epochs have completed.
    """

    expected_epochs = int(
        training_config["epochs"]
    )

    completed_epochs = int(
        state["epoch"]
    )

    if completed_epochs != expected_epochs:
        raise RuntimeError(
            "Training completed with an unexpected "
            f"epoch count: "
            f"{completed_epochs} / {expected_epochs}"
        )

    if state["global_step"] < 0:
        raise RuntimeError(
            "Final global step cannot be negative."
        )

    if state["train_loss"] is None:
        raise RuntimeError(
            "Final training loss is missing."
        )

    if not torch.isfinite(
        torch.tensor(
            float(
                state["train_loss"]
            )
        )
    ):
        raise FloatingPointError(
            "Final training loss is non-finite."
        )

    if (
        state["validation_loss"] is not None
        and not torch.isfinite(
            torch.tensor(
                float(
                    state["validation_loss"]
                )
            )
        )
    ):
        raise FloatingPointError(
            "Final validation loss is non-finite."
        )

    model.eval()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Epochs Completed : "
        f"{completed_epochs}"
    )

    print(
        f"Global Steps     : "
        f"{state['global_step']:,}"
    )

    print(
        f"Final Train Loss : "
        f"{state['train_loss']:.6f}"
    )

    if state["validation_loss"] is not None:

        print(
            f"Final Val Loss   : "
            f"{state['validation_loss']:.6f}"
        )

    if state["best_metric"] is not None:

        print(
            f"Best Val Loss    : "
            f"{state['best_metric']:.6f}"
        )

    print("=" * 80)

    return state


class TokenizedTorchDataset(
    Dataset[dict[str, torch.Tensor]]
):
    """
    PyTorch Dataset wrapper for the tokenized financial dataset.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
    ) -> None:
        """
        Initialize the dataset from tokenized samples.
        """

        if not isinstance(
            samples,
            list,
        ):
            raise TypeError(
                "samples must be a list."
            )

        if not samples:
            raise ValueError(
                "Tokenized dataset split cannot be empty."
            )

        required_keys = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        for index, sample in enumerate(
            samples
        ):

            if not isinstance(
                sample,
                dict,
            ):
                raise TypeError(
                    f"Sample {index} must be a dictionary."
                )

            missing_keys = sorted(
                required_keys
                - set(sample.keys())
            )

            if missing_keys:
                raise RuntimeError(
                    f"Sample {index} is missing "
                    f"required fields: {missing_keys}"
                )

        self.samples = samples

    def __len__(
        self,
    ) -> int:
        """
        Return the number of samples.
        """

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        """
        Return one sample as PyTorch tensors.
        """

        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "Dataset index must be an integer."
            )

        if index < 0 or index >= len(
            self.samples
        ):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        sample = self.samples[
            index
        ]

        input_ids = torch.tensor(
            sample["input_ids"],
            dtype=torch.long,
        )

        attention_mask = torch.tensor(
            sample["attention_mask"],
            dtype=torch.long,
        )

        labels = torch.tensor(
            sample["labels"],
            dtype=torch.long,
        )

        if (
            input_ids.shape
            != attention_mask.shape
        ):
            raise RuntimeError(
                "input_ids and attention_mask "
                "have different shapes."
            )

        if (
            input_ids.shape
            != labels.shape
        ):
            raise RuntimeError(
                "input_ids and labels "
                "have different shapes."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _build_torch_datasets(
    tokenized_dataset: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[
    str,
    TokenizedTorchDataset,
]:
    """
    Convert every tokenized split into a PyTorch Dataset.
    """

    if not isinstance(
        tokenized_dataset,
        dict,
    ):
        raise TypeError(
            "tokenized_dataset must be a dictionary."
        )

    if "train" not in tokenized_dataset:
        raise RuntimeError(
            "Tokenized dataset must contain a train split."
        )

    torch_datasets: dict[
        str,
        TokenizedTorchDataset,
    ] = {}

    for split_name, samples in (
        tokenized_dataset.items()
    ):

        torch_datasets[
            split_name
        ] = TokenizedTorchDataset(
            samples
        )

    if len(
        torch_datasets["train"]
    ) == 0:
        raise RuntimeError(
            "Training dataset contains no samples."
        )

    return torch_datasets


def _build_training_dataloaders(
    torch_datasets: dict[
        str,
        TokenizedTorchDataset,
    ],
    training_config: dict[str, Any],
) -> tuple[
    DataLoader[Any],
    DataLoader[Any] | None,
]:
    """
    Build training and validation DataLoaders.
    """

    if "train" not in torch_datasets:
        raise RuntimeError(
            "Training dataset is missing."
        )

    train_dataloader = _build_dataloader(
        dataset=torch_datasets[
            "train"
        ],
        training_config=training_config,
        split_name="train",
    )

    validation_dataloader = None

    validation_split_name: str | None = None

    if "validation" in torch_datasets:
        validation_split_name = "validation"

    elif "dev" in torch_datasets:
        validation_split_name = "dev"

    if (
        training_config[
            "validation"
        ]["enabled"]
        and validation_split_name is not None
    ):

        validation_dataloader = _build_dataloader(
            dataset=torch_datasets[
                validation_split_name
            ],
            training_config=training_config,
            split_name=validation_split_name,
        )

    return (
        train_dataloader,
        validation_dataloader,
    )

def _build_training_components(
    training_config: dict[str, Any],
    dataloader_length: int,
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """
    Build the model, optimizer, and scheduler using the
    project's dedicated training modules.

    The scheduler receives the actual training DataLoader length
    and gradient accumulation factor so it can calculate the
    number of optimizer update steps correctly.
    """

    if not isinstance(
        training_config,
        dict,
    ):
        raise TypeError(
            "training_config must be a dictionary."
        )

    if not isinstance(
        dataloader_length,
        int,
    ) or dataloader_length <= 0:
        raise ValueError(
            "dataloader_length must be a positive integer."
        )

    gradient_accumulation_steps = training_config[
        "gradient_accumulation_steps"
    ]

    if not isinstance(
        gradient_accumulation_steps,
        int,
    ) or gradient_accumulation_steps <= 0:
        raise ValueError(
            "training.gradient_accumulation_steps must be "
            "a positive integer."
        )

    from src.training.model import (
        get_model,
    )

    from src.training.optimizer import (
        get_optimizer,
    )

    from src.training.scheduler import (
        get_scheduler,
    )

    model = get_model()

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "get_model() must return a torch.nn.Module."
        )

    optimizer = get_optimizer(
        model=model,
    )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "get_optimizer() must return "
            "a torch.optim.Optimizer."
        )

    scheduler = get_scheduler(
        optimizer=optimizer,
        dataloader_length=dataloader_length,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
    )

    if not isinstance(
        scheduler,
        torch.optim.lr_scheduler.LRScheduler,
    ):
        raise TypeError(
            "get_scheduler() must return "
            "a torch.optim.lr_scheduler.LRScheduler."
        )

    return (
        model,
        optimizer,
        scheduler,
    )


def _get_training_device(
    model: torch.nn.Module,
) -> torch.device:
    """
    Determine the device on which the model is currently loaded.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    try:

        first_parameter = next(
            model.parameters()
        )

    except StopIteration as exc:

        raise RuntimeError(
            "Model contains no parameters."
        ) from exc

    device = first_parameter.device

    if device.type == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Model is assigned to CUDA but CUDA "
                "is not available."
            )

        if device.index is not None:

            device_count = torch.cuda.device_count()

            if device.index >= device_count:
                raise RuntimeError(
                    "Model is assigned to CUDA device "
                    f"{device.index}, but only "
                    f"{device_count} CUDA devices exist."
                )

    return device


def _validate_training_components(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """
    Validate the model, optimizer, and scheduler before
    entering the training loop.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "optimizer must be a torch.optim.Optimizer."
        )

    if not isinstance(
        scheduler,
        torch.optim.lr_scheduler.LRScheduler,
    ):
        raise TypeError(
            "scheduler must be a PyTorch learning-rate scheduler."
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }

    trainable_parameter_ids = {
        id(parameter)
        for parameter in trainable_parameters
    }

    missing_parameters = (
        trainable_parameter_ids
        - optimizer_parameter_ids
    )

    if missing_parameters:
        raise RuntimeError(
            "Optimizer does not contain all trainable "
            "model parameters."
        )

    if not optimizer.param_groups:
        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    if not scheduler.optimizer is optimizer:
        raise RuntimeError(
            "Scheduler is not attached to the supplied optimizer."
        )

    print("=" * 80)
    print("TRAINING COMPONENT VALIDATION")
    print("=" * 80)

    print(
        f"Model Type       : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Optimizer Type   : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Scheduler Type   : "
        f"{scheduler.__class__.__name__}"
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"Trainable Params : "
        f"{trainable_parameter_count:,}"
    )

    print(
        "Status           : PASSED"
    )

    print("=" * 80)


def _prepare_training_pipeline(
    training_config: dict[str, Any],
    dataloader_length: int,
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.device,
]:
    """
    Build and validate all components required by the trainer.
    """

    if not isinstance(
        dataloader_length,
        int,
    ) or dataloader_length <= 0:
        raise ValueError(
            "dataloader_length must be a positive integer."
        )

    (
        model,
        optimizer,
        scheduler,
    ) = _build_training_components(
        training_config=training_config,
        dataloader_length=dataloader_length,
    )

    _validate_training_components(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    device = _get_training_device(
        model=model,
    )

    print(
        f"Training Device  : {device}"
    )

    return (
        model,
        optimizer,
        scheduler,
        device,
    )

def train(
    tokenized_dataset: dict[
        str,
        list[dict[str, Any]],
    ],
) -> TrainingState:
    """
    Execute the complete QLoRA training pipeline.

    The trainer consumes already-tokenized dataset splits and
    coordinates model construction, optimizer construction,
    scheduler construction, checkpoint resumption, training,
    validation, best-checkpoint handling, and finalization.

    Evaluation and inference are intentionally outside this
    function and are implemented by their dedicated modules.
    """

    if not isinstance(
        tokenized_dataset,
        dict,
    ):
        raise TypeError(
            "tokenized_dataset must be a dictionary."
        )

    full_config = load_configs()

    if not isinstance(
        full_config,
        dict,
    ):
        raise RuntimeError(
            "Loaded project configuration must be a dictionary."
        )

    training_config = full_config.get(
        "training"
    )

    if not isinstance(
        training_config,
        dict,
    ):
        raise RuntimeError(
            "Project configuration does not contain "
            "a valid training configuration."
        )

    _validate_training_config(
        training_config
    )

    torch_datasets = _build_torch_datasets(
        tokenized_dataset
    )

    (
        train_dataloader,
        validation_dataloader,
    ) = _build_training_dataloaders(
        torch_datasets=torch_datasets,
        training_config=training_config,
    )

    (
        model,
        optimizer,
        scheduler,
        device,
    ) = _prepare_training_pipeline(
        training_config=training_config,
        dataloader_length=len(train_dataloader),
    )

    state = _resume_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        full_config=full_config,
    )

    state = _train_all_epochs(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        training_config=training_config,
        full_config=full_config,
        state=state,
    )

    state = _finalize_training(
        model=model,
        state=state,
        training_config=training_config,
    )

    return state




def main() -> None:
    """
    Run an isolated trainer subsystem validation.

    This validates trainer configuration, dataset conversion,
    DataLoader construction, batch preparation, loss validation,
    gradient-accumulation decisions, and training-state logic
    without launching the production QLoRA training run.
    """

    print("=" * 80)
    print("TRAINER MODULE TEST")
    print("=" * 80)

    training_config = _get_training_config()

    _validate_training_config(
        training_config
    )

    print(
        "Configuration          : PASSED"
    )

    test_samples = [
        {
            "input_ids": [
                1,
                2,
                3,
                4,
            ],
            "attention_mask": [
                1,
                1,
                1,
                1,
            ],
            "labels": [
                1,
                2,
                3,
                4,
            ],
        },
        {
            "input_ids": [
                5,
                6,
                7,
                8,
            ],
            "attention_mask": [
                1,
                1,
                1,
                1,
            ],
            "labels": [
                5,
                6,
                7,
                8,
            ],
        },
        {
            "input_ids": [
                9,
                10,
                11,
                12,
            ],
            "attention_mask": [
                1,
                1,
                1,
                1,
            ],
            "labels": [
                9,
                10,
                11,
                12,
            ],
        },
    ]

    tokenized_dataset = {
        "train": test_samples,
        "dev": test_samples,
    }

    torch_datasets = _build_torch_datasets(
        tokenized_dataset
    )

    if len(
        torch_datasets["train"]
    ) != 3:
        raise RuntimeError(
            "Trainer dataset construction failed."
        )

    print(
        "Dataset Conversion      : PASSED"
    )

    (
        train_dataloader,
        validation_dataloader,
    ) = _build_training_dataloaders(
        torch_datasets=torch_datasets,
        training_config=training_config,
    )

    if len(
        train_dataloader
    ) == 0:
        raise RuntimeError(
            "Training DataLoader is empty."
        )

    if validation_dataloader is None:
        raise RuntimeError(
            "Validation DataLoader was not created."
        )

    print(
        "DataLoader Construction : PASSED"
    )

    first_batch = next(
        iter(train_dataloader)
    )

    prepared_batch = _prepare_batch(
        batch=first_batch,
        device=torch.device("cpu"),
    )

    if prepared_batch[
        "input_ids"
    ].shape != prepared_batch[
        "labels"
    ].shape:
        raise RuntimeError(
            "Prepared batch tensor shapes are inconsistent."
        )

    print(
        "Batch Preparation       : PASSED"
    )

    if not _should_optimizer_step(
        batch_index=3,
        total_batches=4,
        gradient_accumulation_steps=4,
    ):
        raise RuntimeError(
            "Full accumulation boundary was not detected."
        )

    if not _should_optimizer_step(
        batch_index=4,
        total_batches=5,
        gradient_accumulation_steps=4,
    ):
        raise RuntimeError(
            "Final partial accumulation window was not detected."
        )

    if _should_optimizer_step(
        batch_index=1,
        total_batches=5,
        gradient_accumulation_steps=4,
    ):
        raise RuntimeError(
            "Premature optimizer step was detected."
        )

    print(
        "Gradient Accumulation   : PASSED"
    )

    state = _create_training_state()

    state = _update_training_state(
        state=state,
        epoch=1,
        global_step=2,
        train_loss=1.25,
        validation_loss=1.10,
        learning_rate=0.0001,
        gradient_norm=0.75,
    )

    if state["epoch"] != 1:
        raise RuntimeError(
            "Training state epoch update failed."
        )

    if state["global_step"] != 2:
        raise RuntimeError(
            "Training state global-step update failed."
        )

    if state["best_metric"] is not None:
        raise RuntimeError(
            "Fresh training state should not have "
            "a best metric before best-metric evaluation."
        )

    if not _is_best_validation_loss(
        validation_loss=1.10,
        best_metric=None,
    ):
        raise RuntimeError(
            "Initial validation loss should be considered best."
        )

    if not _is_best_validation_loss(
        validation_loss=1.10,
        best_metric=1.20,
    ):
        raise RuntimeError(
            "Improved validation loss was not detected."
        )

    if _is_best_validation_loss(
        validation_loss=1.30,
        best_metric=1.20,
    ):
        raise RuntimeError(
            "Worsened validation loss was incorrectly "
            "considered an improvement."
        )

    print(
        "Training State Logic    : PASSED"
    )

    print(
        "Status                  : PASSED"
    )

    print("=" * 80)
    print(
        "Trainer module is ready for training integration."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()


