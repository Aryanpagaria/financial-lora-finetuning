from __future__ import annotations

import copy
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from src.utils.config_loader import load_configs

CheckpointState = dict[str, Any]


def _get_checkpoint_config() -> dict[str, Any]:
    """
    Load and validate the checkpoint configuration.
    """

    config = load_configs()

    if "checkpoint" not in config:
        raise RuntimeError(
            "Checkpoint configuration is missing."
        )

    checkpoint_config = config["checkpoint"]

    required_keys = {
        "root_directory",
        "latest_directory",
        "best_directory",
        "lora_export_directory",
        "state_directory",
        "save_strategy",
        "resume",
        "best_model",
        "retention",
    }

    missing_keys = sorted(
        required_keys
        - set(checkpoint_config.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Missing checkpoint configuration keys: "
            f"{missing_keys}"
        )

    return checkpoint_config


def _create_checkpoint_directories(
    checkpoint_config: dict[str, Any],
) -> dict[str, Path]:
    """
    Create and return all directories required by
    the checkpoint system.
    """

    directory_keys = {
        "root": "root_directory",
        "latest": "latest_directory",
        "best": "best_directory",
        "lora_export": "lora_export_directory",
        "state": "state_directory",
    }

    directories: dict[str, Path] = {}

    for directory_name, config_key in directory_keys.items():

        configured_path = checkpoint_config.get(
            config_key
        )

        if not isinstance(
            configured_path,
            str,
        ):
            raise RuntimeError(
                f"Checkpoint configuration "
                f"'{config_key}' must be a string."
            )

        if not configured_path.strip():
            raise RuntimeError(
                f"Checkpoint configuration "
                f"'{config_key}' cannot be empty."
            )

        directory = Path(
            configured_path
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        directories[
            directory_name
        ] = directory

    return directories


def _capture_rng_state() -> dict[str, Any]:
    """
    Capture Python, NumPy, PyTorch CPU, and CUDA random
    number generator states.

    These states are required for deterministic checkpoint
    resumption as closely as the runtime permits.
    """

    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        rng_state[
            "cuda"
        ] = torch.cuda.get_rng_state_all()

    else:

        rng_state[
            "cuda"
        ] = None

    return rng_state


def _restore_rng_state(
    rng_state: dict[str, Any],
) -> None:
    """
    Restore Python, NumPy, PyTorch CPU, and CUDA
    random number generator states.
    """

    if not isinstance(
        rng_state,
        dict,
    ):
        raise TypeError(
            "rng_state must be a dictionary."
        )

    required_keys = {
        "python",
        "numpy",
        "torch",
        "cuda",
    }

    missing_keys = (
        required_keys
        - set(rng_state.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Checkpoint RNG state is missing keys: "
            f"{sorted(missing_keys)}"
        )

    random.setstate(
        rng_state["python"]
    )

    np.random.set_state(
        rng_state["numpy"]
    )

    torch.set_rng_state(
        rng_state["torch"]
    )

    cuda_state = rng_state["cuda"]

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_state
        )

def _validate_rng_determinism() -> None:
    """
    Verify that restoring a captured RNG state reproduces
    exactly the same subsequent random values.

    This validates Python, NumPy, PyTorch CPU, and CUDA RNG
    streams when CUDA is available.
    """

    python_seed = 123456789
    numpy_seed = 987654321
    torch_seed = 246813579

    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    rng_state = _capture_rng_state()

    expected_python = [
        random.random()
        for _ in range(5)
    ]

    expected_numpy = np.random.random(
        5
    )

    expected_torch = torch.rand(
        5,
        dtype=torch.float32,
    )

    expected_cuda = None

    if torch.cuda.is_available():
        expected_cuda = torch.rand(
            5,
            dtype=torch.float32,
            device="cuda",
        ).cpu()

    _restore_rng_state(
        rng_state
    )

    restored_python = [
        random.random()
        for _ in range(5)
    ]

    restored_numpy = np.random.random(
        5
    )

    restored_torch = torch.rand(
        5,
        dtype=torch.float32,
    )

    restored_cuda = None

    if torch.cuda.is_available():
        restored_cuda = torch.rand(
            5,
            dtype=torch.float32,
            device="cuda",
        ).cpu()

    if expected_python != restored_python:
        raise RuntimeError(
            "Python RNG restoration is not deterministic."
        )

    if not np.array_equal(
        expected_numpy,
        restored_numpy,
    ):
        raise RuntimeError(
            "NumPy RNG restoration is not deterministic."
        )

    if not torch.equal(
        expected_torch,
        restored_torch,
    ):
        raise RuntimeError(
            "PyTorch CPU RNG restoration is not deterministic."
        )

    if torch.cuda.is_available():

        if expected_cuda is None:
            raise RuntimeError(
                "CUDA RNG test expected a CUDA tensor."
            )

        if restored_cuda is None:
            raise RuntimeError(
                "CUDA RNG restoration did not produce a tensor."
            )

        if not torch.equal(
            expected_cuda,
            restored_cuda,
        ):
            raise RuntimeError(
                "CUDA RNG restoration is not deterministic."
            )


def _build_checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    training_config: dict[str, Any],
) -> CheckpointState:
    """
    Build a resumable QLoRA training checkpoint.

    Only trainable LoRA adapter parameters are persisted for
    the model state. The frozen quantized base model is loaded
    again from the configured base-model identifier when
    training resumes.
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

    if epoch < 0:
        raise ValueError(
            "epoch cannot be negative."
        )

    if global_step < 0:
        raise ValueError(
            "global_step cannot be negative."
        )

    if not isinstance(
        training_config,
        dict,
    ):
        raise TypeError(
            "training_config must be a dictionary."
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

    lora_state_dict = (
        get_peft_model_state_dict(
            model
        )
    )

    if not lora_state_dict:
        raise RuntimeError(
            "LoRA state dictionary is empty. "
            "The model does not appear to contain "
            "trainable LoRA parameters."
        )

    checkpoint_state: CheckpointState = {
        "format_version": 2,
        "checkpoint_type": "qlora_training",
        "model_type": "lora_adapter_only",
        "model_state_dict": lora_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": (
            None
            if best_metric is None
            else float(best_metric)
        ),
        "rng_state": _capture_rng_state(),
        "training_config": training_config,
    }

    return checkpoint_state





def _validate_checkpoint_state(
    checkpoint_state: CheckpointState,
) -> None:
    """
    Validate a QLoRA training checkpoint.
    """

    if not isinstance(
        checkpoint_state,
        dict,
    ):
        raise TypeError(
            "checkpoint_state must be a dictionary."
        )

    required_keys = {
        "format_version",
        "checkpoint_type",
        "model_type",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "global_step",
        "best_metric",
        "rng_state",
        "training_config",
    }

    missing_keys = sorted(
        required_keys
        - set(checkpoint_state.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required fields: "
            f"{missing_keys}"
        )

    if checkpoint_state[
        "format_version"
    ] != 2:
        raise RuntimeError(
            "Unsupported checkpoint format version: "
            f"{checkpoint_state['format_version']}"
        )

    if checkpoint_state[
        "checkpoint_type"
    ] != "qlora_training":
        raise RuntimeError(
            "Unsupported checkpoint type: "
            f"{checkpoint_state['checkpoint_type']}"
        )

    if checkpoint_state[
        "model_type"
    ] != "lora_adapter_only":
        raise RuntimeError(
            "Checkpoint must contain LoRA adapter "
            "parameters only."
        )

    if not isinstance(
        checkpoint_state[
            "model_state_dict"
        ],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint LoRA state is invalid."
        )

    if not checkpoint_state[
        "model_state_dict"
    ]:
        raise RuntimeError(
            "Checkpoint LoRA state is empty."
        )

    if not isinstance(
        checkpoint_state[
            "optimizer_state_dict"
        ],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint optimizer state is invalid."
        )

    if not isinstance(
        checkpoint_state[
            "scheduler_state_dict"
        ],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint scheduler state is invalid."
        )

    epoch = checkpoint_state[
        "epoch"
    ]

    if not isinstance(
        epoch,
        int,
    ) or epoch < 0:
        raise RuntimeError(
            "Checkpoint epoch must be a "
            "non-negative integer."
        )

    global_step = checkpoint_state[
        "global_step"
    ]

    if not isinstance(
        global_step,
        int,
    ) or global_step < 0:
        raise RuntimeError(
            "Checkpoint global_step must be a "
            "non-negative integer."
        )

    best_metric = checkpoint_state[
        "best_metric"
    ]

    if best_metric is not None:

        if not isinstance(
            best_metric,
            (int, float),
        ):
            raise RuntimeError(
                "Checkpoint best_metric must be "
                "numeric or None."
            )

        if not torch.isfinite(
            torch.tensor(
                float(best_metric)
            )
        ):
            raise RuntimeError(
                "Checkpoint best_metric must be finite."
            )

    if not isinstance(
        checkpoint_state[
            "rng_state"
        ],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint RNG state is invalid."
        )

    if not isinstance(
        checkpoint_state[
            "training_config"
        ],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint training configuration is invalid."
        )


def _save_checkpoint_file(
    checkpoint_state: CheckpointState,
    checkpoint_path: Path,
) -> None:
    """
    Atomically save checkpoint state to disk.

    A temporary file is written first and then moved into
    place so an interrupted write does not leave a partially
    written checkpoint at the final path.
    """

    _validate_checkpoint_state(
        checkpoint_state
    )

    if not isinstance(
        checkpoint_path,
        Path,
    ):
        raise TypeError(
            "checkpoint_path must be a pathlib.Path."
        )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    try:

        torch.save(
            checkpoint_state,
            temporary_path,
        )

        temporary_path.replace(
            checkpoint_path
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise


def _load_checkpoint_file(
    checkpoint_path: Path,
) -> CheckpointState:
    """
    Safely load and validate a checkpoint from disk.
    """

    if not isinstance(
        checkpoint_path,
        Path,
    ):
        raise TypeError(
            "checkpoint_path must be a pathlib.Path."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    try:

        checkpoint_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as error:

        raise RuntimeError(
            f"Failed to load checkpoint: "
            f"{checkpoint_path}"
        ) from error

    _validate_checkpoint_state(
        checkpoint_state
    )

    return checkpoint_state


def _get_checkpoint_candidates(
    checkpoint_directory: Path,
) -> list[Path]:
    """
    Return valid checkpoint files from a directory.

    Checkpoints are expected to use the .pt extension.
    Temporary files created during atomic writes are ignored.
    """

    if not isinstance(
        checkpoint_directory,
        Path,
    ):
        raise TypeError(
            "checkpoint_directory must be a pathlib.Path."
        )

    if not checkpoint_directory.exists():
        return []

    if not checkpoint_directory.is_dir():
        raise RuntimeError(
            "Checkpoint directory is not a directory: "
            f"{checkpoint_directory}"
        )

    candidates = [
        path
        for path in checkpoint_directory.glob("*.pt")
        if path.is_file()
        and not path.name.endswith(".tmp")
    ]

    candidates.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates


def _find_latest_checkpoint(
    checkpoint_directory: Path,
) -> Path | None:
    """
    Locate the configured latest checkpoint.

    The latest checkpoint is stored as latest.pt.
    """

    if not isinstance(
        checkpoint_directory,
        Path,
    ):
        raise TypeError(
            "checkpoint_directory must be a pathlib.Path."
        )

    if not checkpoint_directory.exists():
        return None

    if not checkpoint_directory.is_dir():
        raise RuntimeError(
            "Checkpoint directory is not a directory: "
            f"{checkpoint_directory}"
        )

    latest_path = (
        checkpoint_directory
        / "latest.pt"
    )

    if not latest_path.exists():
        return None

    if not latest_path.is_file():
        raise RuntimeError(
            f"Latest checkpoint path is not a file: "
            f"{latest_path}"
        )

    try:

        _load_checkpoint_file(
            latest_path
        )

    except Exception as error:

        raise RuntimeError(
            "The latest checkpoint exists but failed "
            "validation: "
            f"{latest_path}"
        ) from error

    return latest_path






def _save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    training_config: dict[str, Any],
    checkpoint_path: Path,
) -> Path:
    """
    Build and save a complete training checkpoint.

    The checkpoint contains everything required by the
    trainer to resume the training process.
    """

    checkpoint_state = _build_checkpoint_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        training_config=training_config,
    )

    _save_checkpoint_file(
        checkpoint_state=checkpoint_state,
        checkpoint_path=checkpoint_path,
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Checkpoint save reported success, but the "
            f"checkpoint does not exist: {checkpoint_path}"
        )

    return checkpoint_path



def _load_training_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    """
    Restore a QLoRA training checkpoint.

    The base quantized model is expected to already have been
    constructed by model.py. Only the saved LoRA adapter state
    is restored into that model.
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

    checkpoint_state = _load_checkpoint_file(
        checkpoint_path
    )

    lora_state_dict = checkpoint_state[
        "model_state_dict"
    ]

    try:

        load_result = (
            set_peft_model_state_dict(
                model,
                lora_state_dict,
            )
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to restore LoRA adapter state "
            "from checkpoint."
        ) from error

    if load_result is not None:

        unexpected_keys = getattr(
            load_result,
            "unexpected_keys",
            [],
        )

        if unexpected_keys:

            raise RuntimeError(
                "Checkpoint contains unexpected LoRA "
                f"parameters: {unexpected_keys}"
            )

    try:

        optimizer.load_state_dict(
            checkpoint_state[
                "optimizer_state_dict"
            ]
        )

    except Exception as error:

        raise RuntimeError(
            "Checkpoint optimizer state is incompatible "
            "with the current optimizer."
        ) from error

    try:

        scheduler.load_state_dict(
            checkpoint_state[
                "scheduler_state_dict"
            ]
        )

    except Exception as error:

        raise RuntimeError(
            "Checkpoint scheduler state is incompatible "
            "with the current scheduler."
        ) from error

    _restore_rng_state(
        checkpoint_state[
            "rng_state"
        ]
    )

    return {
        "epoch": checkpoint_state[
            "epoch"
        ],
        "global_step": checkpoint_state[
            "global_step"
        ],
        "best_metric": checkpoint_state[
            "best_metric"
        ],
        "training_config": checkpoint_state[
            "training_config"
        ],
    }


def _validate_optimizer_state_restoration(
    original_optimizer: torch.optim.Optimizer,
    restored_optimizer: torch.optim.Optimizer,
) -> None:
    """
    Verify that optimizer state was restored exactly.

    The comparison covers optimizer parameter groups and
    tensor/scalar optimizer state values.
    """

    original_state = original_optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()

    original_groups = original_state[
        "param_groups"
    ]

    restored_groups = restored_state[
        "param_groups"
    ]

    if len(original_groups) != len(
        restored_groups
    ):
        raise RuntimeError(
            "Restored optimizer has a different number "
            "of parameter groups."
        )

    for group_index, (
        original_group,
        restored_group,
    ) in enumerate(
        zip(
            original_groups,
            restored_groups,
        )
    ):

        if original_group.keys() != restored_group.keys():
            raise RuntimeError(
                "Optimizer parameter-group structure "
                f"differs for group {group_index}."
            )

        for key in original_group:

            original_value = original_group[key]
            restored_value = restored_group[key]

            if isinstance(
                original_value,
                torch.Tensor,
            ):

                if not torch.equal(
                    original_value,
                    restored_value,
                ):
                    raise RuntimeError(
                        "Optimizer parameter-group tensor "
                        f"state differs for key '{key}'."
                    )

            elif original_value != restored_value:

                raise RuntimeError(
                    "Optimizer parameter-group value "
                    f"differs for key '{key}'."
                )

    original_optimizer_state = original_state[
        "state"
    ]

    restored_optimizer_state = restored_state[
        "state"
    ]

    if original_optimizer_state.keys() != restored_optimizer_state.keys():
        raise RuntimeError(
            "Restored optimizer state parameter IDs differ."
        )

    for parameter_id in original_optimizer_state:

        original_parameter_state = (
            original_optimizer_state[
                parameter_id
            ]
        )

        restored_parameter_state = (
            restored_optimizer_state[
                parameter_id
            ]
        )

        if (
            original_parameter_state.keys()
            != restored_parameter_state.keys()
        ):
            raise RuntimeError(
                "Optimizer state entries differ for "
                f"parameter ID {parameter_id}."
            )

        for state_name in original_parameter_state:

            original_value = (
                original_parameter_state[
                    state_name
                ]
            )

            restored_value = (
                restored_parameter_state[
                    state_name
                ]
            )

            if isinstance(
                original_value,
                torch.Tensor,
            ):

                if not torch.equal(
                    original_value.cpu(),
                    restored_value.cpu(),
                ):
                    raise RuntimeError(
                        "Optimizer tensor state differs "
                        f"for '{state_name}'."
                    )

            else:

                if original_value != restored_value:
                    raise RuntimeError(
                        "Optimizer scalar state differs "
                        f"for '{state_name}'."
                    )

def _validate_scheduler_state_restoration(
    original_scheduler: torch.optim.lr_scheduler.LRScheduler,
    restored_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """
    Verify that scheduler state was restored exactly.

    Scheduler state and optimizer learning rates are both
    compared because a scheduler can maintain internal state
    while the optimizer contains the effective learning rate.
    """

    original_state = original_scheduler.state_dict()
    restored_state = restored_scheduler.state_dict()

    if original_state.keys() != restored_state.keys():
        raise RuntimeError(
            "Restored scheduler state keys differ."
        )

    for key in original_state:

        original_value = original_state[key]
        restored_value = restored_state[key]

        if isinstance(
            original_value,
            torch.Tensor,
        ):

            if not torch.equal(
                original_value,
                restored_value,
            ):
                raise RuntimeError(
                    "Scheduler tensor state differs "
                    f"for '{key}'."
                )

        elif isinstance(
            original_value,
            dict,
        ):

            if original_value != restored_value:
                raise RuntimeError(
                    "Scheduler dictionary state differs "
                    f"for '{key}'."
                )

        elif isinstance(
            original_value,
            list,
        ):

            if original_value != restored_value:
                raise RuntimeError(
                    "Scheduler list state differs "
                    f"for '{key}'."
                )

        elif original_value != restored_value:

            raise RuntimeError(
                "Scheduler state differs "
                f"for '{key}'."
            )

    original_learning_rates = [
        float(group["lr"])
        for group in original_scheduler.optimizer.param_groups
    ]

    restored_learning_rates = [
        float(group["lr"])
        for group in restored_scheduler.optimizer.param_groups
    ]

    if len(original_learning_rates) != len(
        restored_learning_rates
    ):
        raise RuntimeError(
            "Restored scheduler has a different number "
            "of optimizer learning-rate groups."
        )

    for index, (
        original_learning_rate,
        restored_learning_rate,
    ) in enumerate(
        zip(
            original_learning_rates,
            restored_learning_rates,
        )
    ):

        if not math.isclose(
            original_learning_rate,
            restored_learning_rate,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                "Restored learning rate differs for "
                f"parameter group {index}."
            )

def validate_resume_compatibility(
    checkpoint_training_config: dict[str, Any],
    current_training_config: dict[str, Any],
) -> None:
    """
    Validate that the current training configuration is compatible
    with a saved checkpoint.

    Configuration values that affect model semantics, LoRA structure,
    optimizer behavior, scheduler behavior, batching, sequence length,
    precision, or reproducibility must remain unchanged when training
    is resumed.

    Logging, output directories, checkpoint retention, and evaluation
    output are intentionally excluded from compatibility validation.
    """

    if not isinstance(
        checkpoint_training_config,
        dict,
    ):
        raise TypeError(
            "checkpoint_training_config must be a dictionary."
        )

    if not isinstance(
        current_training_config,
        dict,
    ):
        raise TypeError(
            "current_training_config must be a dictionary."
        )

    protected_paths = (
        "model.name",
        "model.quantization.load_in_4bit",
        "model.quantization.quantization_type",
        "model.quantization.compute_dtype",
        "lora.rank",
        "lora.alpha",
        "lora.dropout",
        "lora.bias",
        "lora.task_type",
        "lora.target_modules",
        "training.batch_size",
        "training.gradient_accumulation_steps",
        "training.max_seq_length",
        "training.optimizer",
        "training.scheduler",
        "training.precision.fp16",
        "training.precision.bf16",
        "training.reproducibility.seed",
    )

    def get_nested_value(
        configuration: dict[str, Any],
        path: str,
    ) -> tuple[bool, Any]:
        """
        Resolve a dotted configuration path.
        """

        current: Any = configuration

        for component in path.split("."):

            if not isinstance(
                current,
                dict,
            ):
                return False, None

            if component not in current:
                return False, None

            current = current[
                component
            ]

        return True, current

    def normalize_configuration(
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize legacy flat training configuration into
        the canonical nested configuration structure.

        New checkpoints should use the canonical nested
        structure directly.
        """

        normalized = copy.deepcopy(
            configuration
        )

        training_value = normalized.get(
            "training"
        )

        if isinstance(
            training_value,
            dict,
        ):
            return normalized

        legacy_training_keys = {
            "batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
            "optimizer",
            "scheduler",
            "precision",
            "reproducibility",
            "epochs",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "warmup_ratio",
        }

        training_section: dict[str, Any] = {}

        for key in legacy_training_keys:

            if key in normalized:

                training_section[
                    key
                ] = normalized.pop(
                    key
                )

        if training_section:

            normalized[
                "training"
            ] = training_section

        return normalized

    checkpoint_config = normalize_configuration(
        checkpoint_training_config
    )

    current_config = normalize_configuration(
        current_training_config
    )

    mismatches: list[str] = []

    for path in protected_paths:

        (
            checkpoint_exists,
            checkpoint_value,
        ) = get_nested_value(
            checkpoint_config,
            path,
        )

        (
            current_exists,
            current_value,
        ) = get_nested_value(
            current_config,
            path,
        )

        if not checkpoint_exists:

            raise RuntimeError(
                "Checkpoint training configuration "
                "is missing required protected field: "
                f"{path}"
            )

        if not current_exists:

            mismatches.append(
                f"{path}: missing from current configuration"
            )

            continue

        if checkpoint_value != current_value:

            mismatches.append(
                f"{path}: "
                f"checkpoint={checkpoint_value!r}, "
                f"current={current_value!r}"
            )

    if mismatches:

        raise RuntimeError(
            "Training configuration is incompatible "
            "with the resume checkpoint:\n"
            + "\n".join(
                f"  - {mismatch}"
                for mismatch in mismatches
            )
        )


def _update_latest_checkpoint(
    checkpoint_path: Path,
    latest_directory: Path,
) -> Path:
    """
    Update the latest checkpoint pointer by copying the
    supplied checkpoint into the configured latest directory.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    latest_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        latest_directory
        / "latest.pt"
    )

    temporary_path = latest_path.with_suffix(
        ".pt.tmp"
    )

    try:

        shutil.copy2(
            checkpoint_path,
            temporary_path,
        )

        temporary_path.replace(
            latest_path
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    return latest_path


def _update_best_checkpoint(
    checkpoint_path: Path,
    best_directory: Path,
) -> Path:
    """
    Update the best-model checkpoint.

    The best checkpoint is a complete training checkpoint,
    allowing the trainer to resume from the best validation
    state if required.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    best_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        best_directory
        / "best.pt"
    )

    temporary_path = best_path.with_suffix(
        ".pt.tmp"
    )

    try:

        shutil.copy2(
            checkpoint_path,
            temporary_path,
        )

        temporary_path.replace(
            best_path
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    return best_path


def _cleanup_old_checkpoints(
    checkpoint_directory: Path,
    max_checkpoints: int,
) -> None:
    """
    Remove old epoch checkpoints according to the configured
    retention limit.

    The newest checkpoints are retained.
    """

    if max_checkpoints <= 0:
        raise ValueError(
            "max_checkpoints must be greater than zero."
        )

    candidates = _get_checkpoint_candidates(
        checkpoint_directory
    )

    if len(candidates) <= max_checkpoints:
        return

    checkpoints_to_remove = candidates[
        max_checkpoints:
    ]

    for checkpoint_path in checkpoints_to_remove:

        try:

            checkpoint_path.unlink()

        except OSError as error:

            raise RuntimeError(
                "Failed to remove old checkpoint: "
                f"{checkpoint_path}"
            ) from error


def find_resume_checkpoint() -> Path | None:
    """
    Find the latest valid checkpoint configured for
    automatic training resumption.

    Returns:
        The latest valid checkpoint path, or None when
        no resumable checkpoint exists.
    """

    checkpoint_config = _get_checkpoint_config()

    resume_config = checkpoint_config[
        "resume"
    ]

    if not resume_config["enabled"]:
        return None

    checkpoint_directory = Path(
        resume_config[
            "checkpoint_directory"
        ]
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_checkpoint = (
        _find_latest_checkpoint(
            checkpoint_directory
        )
    )

    if latest_checkpoint is None:
        return None

    return latest_checkpoint



def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    training_config: dict[str, Any],
) -> Path:
    """
    Save a complete training checkpoint and update the
    latest checkpoint reference.

    Returns:
        Path to the saved epoch checkpoint.
    """

    checkpoint_config = _get_checkpoint_config()

    directories = (
        _create_checkpoint_directories(
            checkpoint_config
        )
    )

    checkpoint_directory = directories[
        "root"
    ]

    checkpoint_path = (
        checkpoint_directory
        / f"epoch_{epoch}.pt"
    )

    saved_path = _save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        training_config=training_config,
        checkpoint_path=checkpoint_path,
    )

    _update_latest_checkpoint(
        checkpoint_path=saved_path,
        latest_directory=directories[
            "latest"
        ],
    )

    retention_config = checkpoint_config[
        "retention"
    ]

    if retention_config["enabled"]:

        _cleanup_old_checkpoints(
            checkpoint_directory=(
                checkpoint_directory
            ),
            max_checkpoints=int(
                retention_config[
                    "max_checkpoints"
                ]
            ),
        )

    return saved_path


def resume_from_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    current_training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Restore a complete training state from a checkpoint.

    When current_training_config is provided, protected
    configuration fields are compared before any model,
    optimizer, scheduler, or RNG state is restored.

    Returns:
        Training metadata containing epoch, global step,
        best metric, and saved training configuration.
    """

    checkpoint_config = _get_checkpoint_config()

    resume_config = checkpoint_config[
        "resume"
    ]

    if not isinstance(
        resume_config,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint resume configuration must be a dictionary."
        )

    if not resume_config.get(
        "enabled",
        False,
    ):
        raise RuntimeError(
            "Checkpoint resumption is disabled "
            "in the configuration."
        )

    if not isinstance(
        checkpoint_path,
        Path,
    ):
        checkpoint_path = Path(
            checkpoint_path
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Resume checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Resume checkpoint path is not a file: "
            f"{checkpoint_path}"
        )

    checkpoint_state = _load_checkpoint_file(
        checkpoint_path
    )

    saved_training_config = checkpoint_state[
        "training_config"
    ]

    if current_training_config is not None:
        validate_resume_compatibility(
            checkpoint_training_config=(
                saved_training_config
            ),
            current_training_config=(
                current_training_config
            ),
        )

    state = _load_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    return state


def main() -> None:
    """
    Run a complete isolated checkpoint subsystem test.

    The test uses a tiny local PEFT LoRA model so the checkpoint
    system can be validated without downloading the production
    Qwen model or requiring a CUDA-enabled GPU.

    The test validates:

        1. Configuration loading
        2. Checkpoint directory creation
        3. PEFT model construction
        4. LoRA state extraction
        5. Base-model state exclusion
        6. Python RNG determinism
        7. NumPy RNG determinism
        8. PyTorch CPU RNG determinism
        9. CUDA RNG determinism when CUDA is available
        10. Checkpoint construction
        11. Checkpoint schema validation
        12. Atomic checkpoint serialization
        13. Checkpoint deserialization
        14. LoRA state restoration
        15. Optimizer state restoration
        16. Scheduler state restoration
        17. Epoch restoration
        18. Global-step restoration
        19. Best-metric restoration
        20. Training-configuration restoration
        21. Resume compatibility validation
        22. Incompatible configuration rejection
        23. Latest checkpoint creation
        24. Latest checkpoint discovery
        25. Best checkpoint creation
        26. Best checkpoint validation
        27. Checkpoint retention
        28. Temporary artifact cleanup
    """

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )

    print("=" * 80)
    print("CHECKPOINT MODULE TEST")
    print("=" * 80)

    test_directory: Path | None = None

    try:
        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------

        checkpoint_config = _get_checkpoint_config()

        directories = _create_checkpoint_directories(
            checkpoint_config
        )

        root_directory = directories["root"]

        test_directory = (
            root_directory
            / "module_test"
        )

        if test_directory.exists():
            shutil.rmtree(
                test_directory
            )

        test_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Configuration          : PASSED"
        )

        print(
            "Directory Creation     : PASSED"
        )

        # ------------------------------------------------------------------
        # Tiny local PEFT model
        # ------------------------------------------------------------------

        class TinyCheckpointModel(
            torch.nn.Module
        ):
            """
            Tiny local model used exclusively for checkpoint
            subsystem testing.

            The forward signature intentionally accepts the
            arguments commonly forwarded by PEFT.
            """

            def __init__(
                self,
            ) -> None:

                super().__init__()

                self.proj = torch.nn.Linear(
                    4,
                    4,
                    bias=False,
                )

            def forward(
                self,
                input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                """
                Execute the tiny test model.

                Supports input_ids, attention_mask, and
                inputs_embeds so PEFT can invoke the model
                using its normal forwarding interface.
                """

                del attention_mask
                del kwargs

                if inputs_embeds is not None:
                    inputs = inputs_embeds

                elif input_ids is not None:
                    inputs = input_ids

                else:
                    raise ValueError(
                        "TinyCheckpointModel requires either "
                        "input_ids or inputs_embeds."
                    )

                if inputs.ndim != 2:
                    raise ValueError(
                        "TinyCheckpointModel expects a 2D input "
                        "tensor with shape [batch, features]."
                    )

                if inputs.shape[-1] != 4:
                    raise ValueError(
                        "TinyCheckpointModel expects exactly "
                        "4 input features."
                    )

                return self.proj(
                    inputs
                )

        lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "proj",
            ],
            task_type=(
                TaskType.FEATURE_EXTRACTION
            ),
        )

        model = get_peft_model(
            TinyCheckpointModel(),
            lora_config,
        )

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "Checkpoint test PEFT model contains "
                "no trainable LoRA parameters."
            )

        print(
            "PEFT Model             : PASSED"
        )

        # ------------------------------------------------------------------
        # Verify LoRA parameter structure
        # ------------------------------------------------------------------

        lora_state = get_peft_model_state_dict(
            model
        )

        if not lora_state:
            raise RuntimeError(
                "Checkpoint test PEFT model produced "
                "an empty LoRA state dictionary."
            )

        if not all(
            "lora_" in key
            for key in lora_state.keys()
        ):
            raise RuntimeError(
                "Checkpoint test LoRA state contains "
                "non-LoRA parameters."
            )

        print(
            "LoRA Parameters        : PASSED"
        )

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=2e-4,
            weight_decay=0.01,
        )

        # ------------------------------------------------------------------
        # Scheduler
        # ------------------------------------------------------------------

        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: max(
                    0.0,
                    1.0 - (
                        step / 10.0
                    ),
                ),
            )
        )

        # ------------------------------------------------------------------
        # Training configuration
        # ------------------------------------------------------------------

        training_config = {
            "model": {
                "name": (
                    "Qwen/Qwen2.5-3B-Instruct"
                ),
                "quantization": {
                    "load_in_4bit": True,
                    "quantization_type": "nf4",
                    "compute_dtype": "float16",
                },
            },
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 512,
                "optimizer": "adamw",
                "scheduler": "linear",
                "precision": {
                    "fp16": True,
                    "bf16": False,
                },
                "reproducibility": {
                    "seed": 42,
                },
                "epochs": 10,
                "learning_rate": 0.0002,
            },
        }

        # ------------------------------------------------------------------
        # Generate real optimizer state
        # ------------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        dummy_input = torch.ones(
            (1, 4),
            dtype=torch.float32,
        )

        dummy_attention_mask = torch.ones(
            (1, 4),
            dtype=torch.long,
        )

        output = model(
            input_ids=dummy_input,
            attention_mask=dummy_attention_mask,
        )

        if not isinstance(
            output,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Unexpected output type from checkpoint "
                "test model: "
                f"{type(output).__name__}"
            )

        if output.shape != (
            1,
            4,
        ):
            raise RuntimeError(
                "Unexpected checkpoint test model "
                f"output shape: {output.shape}"
            )

        loss = output.sum()

        if not loss.requires_grad:
            raise RuntimeError(
                "Checkpoint test loss does not "
                "require gradients."
            )

        loss.backward()

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        if not optimizer.state:
            raise RuntimeError(
                "Optimizer state was not created "
                "by the checkpoint test step."
            )

        print(
            "Optimizer State       : PASSED"
        )

        print(
            "Scheduler State       : PASSED"
        )

        # ------------------------------------------------------------------
        # RNG deterministic restoration
        # ------------------------------------------------------------------

        _validate_rng_determinism()

        print(
            "Deterministic RNG Test : PASSED"
        )

        # ------------------------------------------------------------------
        # Build checkpoint
        # ------------------------------------------------------------------

        checkpoint_state = _build_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            global_step=120,
            best_metric=1.2345,
            training_config=training_config,
        )

        # ------------------------------------------------------------------
        # Validate checkpoint structure
        # ------------------------------------------------------------------

        _validate_checkpoint_state(
            checkpoint_state
        )

        if checkpoint_state[
            "format_version"
        ] != 2:
            raise RuntimeError(
                "Checkpoint format version must be 2."
            )

        if checkpoint_state[
            "checkpoint_type"
        ] != "qlora_training":
            raise RuntimeError(
                "Checkpoint type must be "
                "'qlora_training'."
            )

        if checkpoint_state[
            "model_type"
        ] != "lora_adapter_only":
            raise RuntimeError(
                "Checkpoint model type must be "
                "'lora_adapter_only'."
            )

        print(
            "Checkpoint Build       : PASSED"
        )

        print(
            "Checkpoint Validation  : PASSED"
        )

        # ------------------------------------------------------------------
        # Verify LoRA state and base-state exclusion
        # ------------------------------------------------------------------

        model_state_keys = list(
            checkpoint_state[
                "model_state_dict"
            ].keys()
        )

        if not model_state_keys:
            raise RuntimeError(
                "Checkpoint contains no model state."
            )

        if not all(
            "lora_" in key
            for key in model_state_keys
        ):
            raise RuntimeError(
                "Checkpoint model state contains "
                "non-LoRA parameters."
            )

        forbidden_tokens = (
            "quant_state",
            "bitsandbytes",
        )

        for key in model_state_keys:

            if any(
                token in key
                for token in forbidden_tokens
            ):
                raise RuntimeError(
                    "Checkpoint contains forbidden "
                    "quantization/base-model state: "
                    f"{key}"
                )

        print(
            "LoRA State Extraction  : PASSED"
        )

        print(
            "Base State Exclusion   : PASSED"
        )

        # ------------------------------------------------------------------
        # Save checkpoint
        # ------------------------------------------------------------------

        checkpoint_path = (
            test_directory
            / "epoch_3.pt"
        )

        _save_checkpoint_file(
            checkpoint_state=checkpoint_state,
            checkpoint_path=checkpoint_path,
        )

        if not checkpoint_path.exists():
            raise RuntimeError(
                "Checkpoint file was not created."
            )

        if checkpoint_path.stat().st_size <= 0:
            raise RuntimeError(
                "Checkpoint file is empty."
            )

        temporary_checkpoint_path = (
            checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".tmp"
            )
        )

        if temporary_checkpoint_path.exists():
            raise RuntimeError(
                "Temporary checkpoint file remains "
                "after atomic save."
            )

        print(
            "Checkpoint Save        : PASSED"
        )

        # ------------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------------

        loaded_state = _load_checkpoint_file(
            checkpoint_path
        )

        if loaded_state[
            "epoch"
        ] != 3:
            raise RuntimeError(
                "Loaded epoch does not match "
                "the saved epoch."
            )

        if loaded_state[
            "global_step"
        ] != 120:
            raise RuntimeError(
                "Loaded global step does not match "
                "the saved global step."
            )

        if loaded_state[
            "best_metric"
        ] != 1.2345:
            raise RuntimeError(
                "Loaded best metric does not match "
                "the saved best metric."
            )

        if loaded_state[
            "training_config"
        ] != training_config:
            raise RuntimeError(
                "Loaded training configuration "
                "does not match the saved configuration."
            )

        print(
            "Checkpoint Load        : PASSED"
        )

        # ------------------------------------------------------------------
        # Create fresh PEFT model for restoration
        # ------------------------------------------------------------------

        restored_model = get_peft_model(
            TinyCheckpointModel(),
            lora_config,
        )

        restored_trainable_parameters = [
            parameter
            for parameter in restored_model.parameters()
            if parameter.requires_grad
        ]

        if not restored_trainable_parameters:
            raise RuntimeError(
                "Fresh PEFT model contains "
                "no trainable parameters."
            )

        restored_optimizer = torch.optim.AdamW(
            restored_trainable_parameters,
            lr=2e-4,
            weight_decay=0.01,
        )

        restored_scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                restored_optimizer,
                lr_lambda=lambda step: max(
                    0.0,
                    1.0 - (
                        step / 10.0
                    ),
                ),
            )
        )

        # ------------------------------------------------------------------
        # Resume compatibility: compatible and incompatible configurations
        # ------------------------------------------------------------------

        compatible_training_config = copy.deepcopy(
            training_config
        )

        incompatible_training_config = copy.deepcopy(
            training_config
        )

        incompatible_training_config[
            "training"
        ][
            "gradient_accumulation_steps"
        ] = 4

        incompatible_configuration_rejected = False

        try:

            validate_resume_compatibility(
                checkpoint_training_config=training_config,
                current_training_config=incompatible_training_config,
            )

        except RuntimeError:

            incompatible_configuration_rejected = True

        if not incompatible_configuration_rejected:

            raise RuntimeError(
                "Resume compatibility validation "
                "failed to reject an incompatible "
                "training.gradient_accumulation_steps "
                "setting."
            )

        print(
            "Resume Compatibility   : PASSED"
        )

        # ------------------------------------------------------------------
        # Restore complete training state
        # ------------------------------------------------------------------

        restored_metadata = resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            current_training_config=compatible_training_config,
        )

        # ------------------------------------------------------------------
        # Restore complete training state
        # ------------------------------------------------------------------

        restored_metadata = resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            current_training_config=compatible_training_config,
        )

        # ------------------------------------------------------------------
        # Verify training metadata
        # ------------------------------------------------------------------

        if restored_metadata[
            "epoch"
        ] != 3:
            raise RuntimeError(
                "Restored epoch does not "
                "match the checkpoint."
            )

        if restored_metadata[
            "global_step"
        ] != 120:
            raise RuntimeError(
                "Restored global step does not "
                "match the checkpoint."
            )

        if restored_metadata[
            "best_metric"
        ] != 1.2345:
            raise RuntimeError(
                "Restored best metric does not "
                "match the checkpoint."
            )

        if restored_metadata[
            "training_config"
        ] != training_config:
            raise RuntimeError(
                "Restored training configuration "
                "does not match the checkpoint."
            )

        print(
            "Training State Restore : PASSED"
        )

        # ------------------------------------------------------------------
        # Verify LoRA restoration
        # ------------------------------------------------------------------

        original_lora_state = (
            get_peft_model_state_dict(
                model
            )
        )

        restored_lora_state = (
            get_peft_model_state_dict(
                restored_model
            )
        )

        if (
            original_lora_state.keys()
            != restored_lora_state.keys()
        ):
            raise RuntimeError(
                "Restored LoRA parameter keys "
                "do not match."
            )

        for parameter_name in original_lora_state:

            original_tensor = (
                original_lora_state[
                    parameter_name
                ]
                .detach()
                .cpu()
            )

            restored_tensor = (
                restored_lora_state[
                    parameter_name
                ]
                .detach()
                .cpu()
            )

            if not torch.equal(
                original_tensor,
                restored_tensor,
            ):
                raise RuntimeError(
                    "Restored LoRA parameter "
                    "does not exactly match the "
                    f"original parameter: "
                    f"{parameter_name}"
                )

        print(
            "LoRA Restore           : PASSED"
        )

        # ------------------------------------------------------------------
        # Verify optimizer restoration
        # ------------------------------------------------------------------

        _validate_optimizer_state_restoration(
            original_optimizer=optimizer,
            restored_optimizer=restored_optimizer,
        )

        print(
            "Optimizer Restoration  : PASSED"
        )

        # ------------------------------------------------------------------
        # Verify scheduler restoration
        # ------------------------------------------------------------------

        _validate_scheduler_state_restoration(
            original_scheduler=scheduler,
            restored_scheduler=restored_scheduler,
        )

        print(
            "Scheduler Restoration  : PASSED"
        )

        # ------------------------------------------------------------------
        # Verify restored model can execute
        # ------------------------------------------------------------------

        restored_output = restored_model(
            input_ids=dummy_input,
            attention_mask=dummy_attention_mask,
        )

        if not isinstance(
            restored_output,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Restored PEFT model produced an "
                f"unexpected output type: "
                f"{type(restored_output).__name__}"
            )

        if restored_output.shape != (
            1,
            4,
        ):
            raise RuntimeError(
                "Restored PEFT model produced "
                f"unexpected output shape: "
                f"{restored_output.shape}"
            )

        print(
            "Restored Model Forward : PASSED"
        )

        # ------------------------------------------------------------------
        # Latest checkpoint
        # ------------------------------------------------------------------

        latest_directory = (
            test_directory
            / "latest"
        )

        latest_path = _update_latest_checkpoint(
            checkpoint_path=checkpoint_path,
            latest_directory=latest_directory,
        )

        if not latest_path.exists():
            raise RuntimeError(
                "Latest checkpoint was not created."
            )

        discovered_latest = _find_latest_checkpoint(
            latest_directory
        )

        if discovered_latest != latest_path:
            raise RuntimeError(
                "Latest checkpoint discovery "
                "returned the wrong checkpoint."
            )

        latest_state = _load_checkpoint_file(
            latest_path
        )

        if latest_state[
            "epoch"
        ] != 3:
            raise RuntimeError(
                "Latest checkpoint contains "
                "the wrong epoch."
            )

        print(
            "Latest Checkpoint      : PASSED"
        )

        # ------------------------------------------------------------------
        # Best checkpoint
        # ------------------------------------------------------------------

        best_directory = (
            test_directory
            / "best"
        )

        best_path = _update_best_checkpoint(
            checkpoint_path=checkpoint_path,
            best_directory=best_directory,
        )

        if not best_path.exists():
            raise RuntimeError(
                "Best checkpoint was not created."
            )

        best_state = _load_checkpoint_file(
            best_path
        )

        if best_state[
            "best_metric"
        ] != 1.2345:
            raise RuntimeError(
                "Best checkpoint contains "
                "the wrong best metric."
            )

        print(
            "Best Checkpoint        : PASSED"
        )

        # ------------------------------------------------------------------
        # Retention test
        # ------------------------------------------------------------------

        retention_directory = (
            test_directory
            / "retention"
        )

        retention_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for epoch_number in range(
            1,
            6,
        ):

            retention_path = (
                retention_directory
                / f"epoch_{epoch_number}.pt"
            )

            retention_state = (
                _build_checkpoint_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch_number,
                    global_step=epoch_number * 10,
                    best_metric=1.0,
                    training_config=training_config,
                )
            )

            _save_checkpoint_file(
                checkpoint_state=retention_state,
                checkpoint_path=retention_path,
            )

        _cleanup_old_checkpoints(
            checkpoint_directory=retention_directory,
            max_checkpoints=3,
        )

        remaining_checkpoints = (
            _get_checkpoint_candidates(
                retention_directory
            )
        )

        if len(remaining_checkpoints) != 3:
            raise RuntimeError(
                "Retention policy failed. "
                "Expected 3 checkpoints, "
                f"found {len(remaining_checkpoints)}."
            )

        remaining_names = {
            path.name
            for path in remaining_checkpoints
        }

        expected_names = {
            "epoch_3.pt",
            "epoch_4.pt",
            "epoch_5.pt",
        }

        if remaining_names != expected_names:
            raise RuntimeError(
                "Retention policy kept the wrong "
                "checkpoints. "
                f"Expected {expected_names}, "
                f"found {remaining_names}."
            )

        print(
            "Retention Policy      : PASSED"
        )

        # ------------------------------------------------------------------
        # Resume checkpoint discovery test
        # ------------------------------------------------------------------

        configured_resume_directory = Path(
            checkpoint_config[
                "resume"
            ][
                "checkpoint_directory"
            ]
        )

        if configured_resume_directory.exists():
            discovered_resume = (
                _find_latest_checkpoint(
                    configured_resume_directory
                )
            )

            if discovered_resume is not None:
                _load_checkpoint_file(
                    discovered_resume
                )

        print(
            "Resume Discovery       : PASSED"
        )

        # ------------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------------

        print("=" * 80)

        print(
            "CHECKPOINT MODULE TEST PASSED"
        )

        print("=" * 80)

        print(
            "QLoRA checkpoint architecture : VALID"
        )

        print(
            "LoRA-only model state         : VALID"
        )

        print(
            "Optimizer restoration         : VALID"
        )

        print(
            "Scheduler restoration         : VALID"
        )

        print(
            "RNG restoration               : VALID"
        )

        print(
            "Resume compatibility          : VALID"
        )

        print(
            "Latest checkpoint             : VALID"
        )

        print(
            "Best checkpoint               : VALID"
        )

        print(
            "Retention policy              : VALID"
        )

        print(
            "Status                        : READY"
        )

        print("=" * 80)

    finally:

        if (
            test_directory is not None
            and test_directory.exists()
        ):

            shutil.rmtree(
                test_directory
            )

        print(
            "Temporary test artifacts removed."
        )


if __name__ == "__main__":
    main()