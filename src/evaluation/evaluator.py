from __future__ import annotations
from transformers import PreTrainedTokenizerBase
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from peft import set_peft_model_state_dict

from src.data.preprocessing import get_preprocessed_dataset
from src.training.checkpoint import _load_checkpoint_file
from src.training.model import get_model
from src.training.tokenizer import get_tokenizer
from src.utils.config_loader import load_configs


EvaluationResult = dict[str, Any]


@dataclass(frozen=True)
class EvaluationExample:
    """
    Evaluation result for one dataset example.
    """

    example_id: str
    filename: str
    reference: str
    prediction: str
    exact_match: bool | None
    normalized_exact_match: bool | None
    numerical_accuracy: bool | None
    failed: bool
    error: str | None


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Aggregate evaluation metrics.
    """

    split: str
    total_examples: int
    evaluated_examples: int
    failed_examples: int
    exact_match: float | None
    normalized_exact_match: float | None
    numerical_accuracy: float | None
    mean_loss: float | None
    perplexity: float | None


class EvaluationDataset(
    Dataset[dict[str, Any]]
):
    """
    Dataset wrapper around preprocessed evaluation examples.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
    ) -> None:
        if not isinstance(
            samples,
            list,
        ):
            raise TypeError(
                "samples must be a list."
            )

        if not samples:
            raise ValueError(
                "Evaluation split cannot be empty."
            )

        required_keys = {
            "prompt",
            "target",
            "id",
            "filename",
        }

        for index, sample in enumerate(
            samples
        ):
            if not isinstance(
                sample,
                dict,
            ):
                raise TypeError(
                    f"Evaluation sample {index} must be a dictionary."
                )

            missing_keys = sorted(
                required_keys
                - set(sample.keys())
            )

            if missing_keys:
                raise RuntimeError(
                    f"Evaluation sample {index} is missing "
                    f"required fields: {missing_keys}"
                )

            for key in required_keys:
                if not isinstance(
                    sample[key],
                    str,
                ):
                    raise TypeError(
                        f"Evaluation sample {index} field "
                        f"'{key}' must be a string."
                    )

        self.samples = samples

    def __len__(
        self,
    ) -> int:
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
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

        return self.samples[
            index
        ]


def _load_evaluation_config() -> dict[str, Any]:
    """
    Load the complete project configuration and return
    the evaluation section.
    """

    config = load_configs()

    if not isinstance(
        config,
        dict,
    ):
        raise RuntimeError(
            "Loaded project configuration must be a dictionary."
        )

    evaluation_config = config.get(
        "evaluation"
    )

    if not isinstance(
        evaluation_config,
        dict,
    ):
        raise RuntimeError(
            "Evaluation configuration is missing or invalid."
        )

    return evaluation_config


def _validate_evaluation_config(
    evaluation_config: dict[str, Any],
) -> None:
    """
    Validate every configuration field consumed by the evaluator.
    """

    if not isinstance(
        evaluation_config,
        dict,
    ):
        raise TypeError(
            "evaluation_config must be a dictionary."
        )

    required_keys = {
        "enabled",
        "batch_size",
        "max_new_tokens",
        "temperature",
        "do_sample",
        "top_p",
        "metrics",
        "splits",
        "generation",
        "output",
        "reproducibility",
    }

    missing_keys = sorted(
        required_keys
        - set(evaluation_config.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Missing evaluation configuration keys: "
            f"{missing_keys}"
        )

    enabled = evaluation_config[
        "enabled"
    ]

    if not isinstance(
        enabled,
        bool,
    ):
        raise TypeError(
            "evaluation.enabled must be boolean."
        )

    batch_size = evaluation_config[
        "batch_size"
    ]

    if (
        not isinstance(
            batch_size,
            int,
        )
        or isinstance(
            batch_size,
            bool,
        )
        or batch_size <= 0
    ):
        raise ValueError(
            "evaluation.batch_size must be a positive integer."
        )

    if batch_size != 1:
        raise ValueError(
            "evaluation.batch_size must be 1 because the "
            "current decoder-only generation pipeline uses "
            "right-padded prompts and requires deterministic "
            "single-example generation."
        )

    max_new_tokens = evaluation_config[
        "max_new_tokens"
    ]

    if (
        not isinstance(
            max_new_tokens,
            int,
        )
        or isinstance(
            max_new_tokens,
            bool,
        )
        or max_new_tokens <= 0
    ):
        raise ValueError(
            "evaluation.max_new_tokens must be a positive integer."
        )

    temperature = evaluation_config[
        "temperature"
    ]

    if (
        not isinstance(
            temperature,
            (int, float),
        )
        or isinstance(
            temperature,
            bool,
        )
        or not math.isfinite(
            float(temperature)
        )
        or temperature < 0
    ):
        raise ValueError(
            "evaluation.temperature must be a finite non-negative number."
        )

    do_sample = evaluation_config[
        "do_sample"
    ]

    if not isinstance(
        do_sample,
        bool,
    ):
        raise TypeError(
            "evaluation.do_sample must be boolean."
        )

    top_p = evaluation_config[
        "top_p"
    ]

    if (
        not isinstance(
            top_p,
            (int, float),
        )
        or isinstance(
            top_p,
            bool,
        )
        or not math.isfinite(
            float(top_p)
        )
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError(
            "evaluation.top_p must be in the interval (0, 1]."
        )

    metrics = evaluation_config[
        "metrics"
    ]

    if not isinstance(
        metrics,
        dict,
    ):
        raise TypeError(
            "evaluation.metrics must be a dictionary."
        )

    metric_keys = {
        "exact_match",
        "normalized_exact_match",
        "numerical_accuracy",
    }

    missing_metric_keys = sorted(
        metric_keys
        - set(metrics.keys())
    )

    if missing_metric_keys:
        raise RuntimeError(
            "Evaluation metrics configuration is missing: "
            f"{missing_metric_keys}"
        )

    for metric_name in metric_keys:
        if not isinstance(
            metrics[metric_name],
            bool,
        ):
            raise TypeError(
                f"evaluation.metrics.{metric_name} must be boolean."
            )

    splits = evaluation_config[
        "splits"
    ]

    if not isinstance(
        splits,
        dict,
    ):
        raise TypeError(
            "evaluation.splits must be a dictionary."
        )

    for split_name in (
        "validation",
        "test",
    ):
        if split_name not in splits:
            raise RuntimeError(
                f"evaluation.splits.{split_name} is missing."
            )

        if not isinstance(
            splits[split_name],
            bool,
        ):
            raise TypeError(
                f"evaluation.splits.{split_name} must be boolean."
            )

    generation = evaluation_config[
        "generation"
    ]

    if not isinstance(
        generation,
        dict,
    ):
        raise TypeError(
            "evaluation.generation must be a dictionary."
        )

    generation_required_keys = {
        "repetition_penalty",
        "no_repeat_ngram_size",
        "early_stopping",
    }

    missing_generation_keys = sorted(
        generation_required_keys
        - set(generation.keys())
    )

    if missing_generation_keys:
        raise RuntimeError(
            "Evaluation generation configuration is missing: "
            f"{missing_generation_keys}"
        )

    repetition_penalty = generation[
        "repetition_penalty"
    ]

    if (
        not isinstance(
            repetition_penalty,
            (int, float),
        )
        or isinstance(
            repetition_penalty,
            bool,
        )
        or not math.isfinite(
            float(repetition_penalty)
        )
        or repetition_penalty <= 0
    ):
        raise ValueError(
            "evaluation.generation.repetition_penalty must "
            "be greater than zero."
        )

    no_repeat_ngram_size = generation[
        "no_repeat_ngram_size"
    ]

    if (
        not isinstance(
            no_repeat_ngram_size,
            int,
        )
        or isinstance(
            no_repeat_ngram_size,
            bool,
        )
        or no_repeat_ngram_size < 0
    ):
        raise ValueError(
            "evaluation.generation.no_repeat_ngram_size must "
            "be a non-negative integer."
        )

    if not isinstance(
        generation["early_stopping"],
        bool,
    ):
        raise TypeError(
            "evaluation.generation.early_stopping must be boolean."
        )

    output = evaluation_config[
        "output"
    ]

    if not isinstance(
        output,
        dict,
    ):
        raise TypeError(
            "evaluation.output must be a dictionary."
        )

    output_required_keys = {
        "directory",
        "save_predictions",
        "save_metrics",
        "save_failed_examples",
    }

    missing_output_keys = sorted(
        output_required_keys
        - set(output.keys())
    )

    if missing_output_keys:
        raise RuntimeError(
            "Evaluation output configuration is missing: "
            f"{missing_output_keys}"
        )

    if not isinstance(
        output["directory"],
        str,
    ) or not output["directory"].strip():
        raise ValueError(
            "evaluation.output.directory must be a non-empty string."
        )

    for key in (
        "save_predictions",
        "save_metrics",
        "save_failed_examples",
    ):
        if not isinstance(
            output[key],
            bool,
        ):
            raise TypeError(
                f"evaluation.output.{key} must be boolean."
            )

    reproducibility = evaluation_config[
        "reproducibility"
    ]

    if not isinstance(
        reproducibility,
        dict,
    ):
        raise TypeError(
            "evaluation.reproducibility must be a dictionary."
        )

    if "seed" not in reproducibility:
        raise RuntimeError(
            "evaluation.reproducibility.seed is missing."
        )

    seed = reproducibility[
        "seed"
    ]

    if (
        not isinstance(
            seed,
            int,
        )
        or isinstance(
            seed,
            bool,
        )
        or seed < 0
    ):
        raise ValueError(
            "evaluation.reproducibility.seed must be a "
            "non-negative integer."
        )

    if (
        not do_sample
        and float(temperature) != 0.0
    ):
        raise ValueError(
            "temperature must be 0.0 when do_sample is false."
        )


def _set_reproducibility_seed(
    seed: int,
) -> None:
    """
    Configure deterministic random seeds for evaluation.
    """

    if (
        not isinstance(
            seed,
            int,
        )
        or isinstance(
            seed,
            bool,
        )
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer."
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

def _validate_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """
    Validate the Hugging Face tokenizer used by evaluation.

    Both slow and fast tokenizer implementations are valid.
    The common PreTrainedTokenizerBase type is therefore used
    instead of requiring the slower PreTrainedTokenizer class.
    """

    if not isinstance(
        tokenizer,
        PreTrainedTokenizerBase,
    ):
        raise TypeError(
            "tokenizer must be a transformers "
            "PreTrainedTokenizerBase."
        )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define a pad_token_id."
        )

    if tokenizer.eos_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define an eos_token_id."
        )

    if tokenizer.padding_side not in {
        "left",
        "right",
    }:
        raise RuntimeError(
            "Tokenizer padding_side must be either "
            "'left' or 'right'."
        )

    if tokenizer.truncation_side not in {
        "left",
        "right",
    }:
        raise RuntimeError(
            "Tokenizer truncation_side must be either "
            "'left' or 'right'."
        )

    if not hasattr(
        tokenizer,
        "encode",
    ):
        raise RuntimeError(
            "Tokenizer does not provide the required "
            "encode operation."
        )

    if not hasattr(
        tokenizer,
        "decode",
    ):
        raise RuntimeError(
            "Tokenizer does not provide the required "
            "decode operation."
        )

def _build_evaluation_dataloader(
    samples: list[dict[str, Any]],
    batch_size: int,
) -> DataLoader[
    dict[str, Any]
]:
    """
    Build the deterministic evaluation DataLoader.
    """

    dataset = EvaluationDataset(
        samples
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def _resolve_checkpoint_path() -> Path:
    """
    Resolve the preferred trained checkpoint.

    The best checkpoint is preferred because evaluation should
    measure the model selected by validation performance.
    The latest checkpoint is used only when a best checkpoint
    does not exist.
    """

    full_config = load_configs()

    if not isinstance(
        full_config,
        dict,
    ):
        raise RuntimeError(
            "Loaded project configuration must be a dictionary."
        )

    checkpoint_config = full_config.get(
        "checkpoint"
    )

    if not isinstance(
        checkpoint_config,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint configuration is missing."
        )

    best_directory = checkpoint_config.get(
        "best_directory"
    )

    latest_directory = checkpoint_config.get(
        "latest_directory"
    )

    if not isinstance(
        best_directory,
        str,
    ) or not best_directory.strip():
        raise RuntimeError(
            "checkpoint.best_directory must be configured."
        )

    if not isinstance(
        latest_directory,
        str,
    ) or not latest_directory.strip():
        raise RuntimeError(
            "checkpoint.latest_directory must be configured."
        )

    best_path = (
        Path(best_directory)
        / "best.pt"
    )

    latest_path = (
        Path(latest_directory)
        / "latest.pt"
    )

    if best_path.exists():
        if not best_path.is_file():
            raise RuntimeError(
                f"Best checkpoint path is not a file: {best_path}"
            )

        return best_path

    if latest_path.exists():
        if not latest_path.is_file():
            raise RuntimeError(
                f"Latest checkpoint path is not a file: {latest_path}"
            )

        return latest_path

    raise FileNotFoundError(
        "No trained checkpoint is available for evaluation. "
        f"Checked best checkpoint '{best_path}' and "
        f"latest checkpoint '{latest_path}'."
    )


def _load_evaluation_model(
    checkpoint_path: Path,
) -> torch.nn.Module:
    """
    Construct the production QLoRA model and restore only its
    LoRA adapter state from a validated training checkpoint.
    """

    if not isinstance(
        checkpoint_path,
        Path,
    ):
        raise TypeError(
            "checkpoint_path must be a pathlib.Path."
        )

    checkpoint_state = _load_checkpoint_file(
        checkpoint_path
    )

    model = get_model()

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "get_model() must return a torch.nn.Module."
        )

    model_state_dict = checkpoint_state.get(
        "model_state_dict"
    )

    if not isinstance(
        model_state_dict,
        dict,
    ) or not model_state_dict:
        raise RuntimeError(
            "Evaluation checkpoint contains no LoRA model state."
        )

    try:
        load_result = set_peft_model_state_dict(
            model,
            model_state_dict,
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to restore LoRA adapter state "
            "for evaluation."
        ) from error

    if load_result is not None:
        unexpected_keys = getattr(
            load_result,
            "unexpected_keys",
            [],
        )

        if unexpected_keys:
            raise RuntimeError(
                "Evaluation checkpoint contains unexpected "
                f"LoRA parameters: {unexpected_keys}"
            )

    model.eval()

    return model


def _get_model_device(
    model: torch.nn.Module,
) -> torch.device:
    """
    Determine the device of the model's first parameter.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    try:
        parameter = next(
            model.parameters()
        )
    except StopIteration as error:
        raise RuntimeError(
            "Evaluation model contains no parameters."
        ) from error

    device = parameter.device

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Evaluation model is on CUDA but CUDA is unavailable."
        )

    return device


def _normalize_answer(
    text: str,
) -> str:
    """
    Normalize an answer for normalized exact-match evaluation.
    """

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string."
        )

    normalized = text.strip().lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = normalized.strip(
        " \t\n\r.,;:!?\"'`"
    )

    normalized = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        normalized,
    )

    return normalized


def _extract_numeric_values(
    text: str,
) -> list[Decimal]:
    """
    Extract explicit decimal/integer numeric values from text.

    The evaluator deliberately does not execute generated text,
    financial programs, Python expressions, or arbitrary arithmetic.
    """

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string."
        )

    numeric_pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"[-+]?"
        r"(?:"
        r"\d{1,3}(?:,\d{3})+"
        r"|"
        r"\d+"
        r")"
        r"(?:\.\d+)?"
        r"(?:[eE][-+]?\d+)?"
        r"%?"
        r"(?![A-Za-z0-9_])"
    )

    values: list[Decimal] = []

    for match in numeric_pattern.finditer(
        text
    ):
        token = match.group(
            0
        )

        has_percent = token.endswith(
            "%"
        )

        if has_percent:
            token = token[:-1]

        token = token.replace(
            ",",
            "",
        )

        try:
            value = Decimal(
                token
            )
        except InvalidOperation:
            continue

        if has_percent:
            value = value / Decimal(
                "100"
            )

        values.append(
            value
        )

    return values


def _numbers_match(
    reference: Decimal,
    prediction: Decimal,
) -> bool:
    """
    Compare numerical answers using a strict relative/absolute
    tolerance appropriate for generated financial values.
    """

    if not reference.is_finite() or not prediction.is_finite():
        return False

    difference = abs(
        reference
        - prediction
    )

    absolute_tolerance = Decimal(
        "0.000001"
    )

    relative_tolerance = (
        abs(reference)
        * Decimal("0.000001")
    )

    tolerance = max(
        absolute_tolerance,
        relative_tolerance,
    )

    return difference <= tolerance


def _numerical_accuracy(
    reference: str,
    prediction: str,
) -> bool:
    """
    Determine numerical correctness without executing generated text.

    A numerical match requires the same number of explicit numeric
    values in the reference and prediction and each corresponding
    value must satisfy the strict numeric tolerance.
    """

    reference_values = _extract_numeric_values(
        reference
    )

    prediction_values = _extract_numeric_values(
        prediction
    )

    if not reference_values:
        return False

    if len(reference_values) != len(
        prediction_values
    ):
        return False

    return all(
        _numbers_match(
            reference_value,
            prediction_value,
        )
        for reference_value, prediction_value
        in zip(
            reference_values,
            prediction_values,
        )
    )


def _calculate_example_metrics(
    reference: str,
    prediction: str,
    evaluation_config: dict[str, Any],
) -> tuple[
    bool | None,
    bool | None,
    bool | None,
]:
    """
    Calculate all enabled per-example metrics.
    """

    metrics_config = evaluation_config[
        "metrics"
    ]

    exact_match = None
    normalized_exact_match = None
    numerical_accuracy = None

    if metrics_config[
        "exact_match"
    ]:
        exact_match = (
            reference.strip()
            == prediction.strip()
        )

    if metrics_config[
        "normalized_exact_match"
    ]:
        normalized_exact_match = (
            _normalize_answer(
                reference
            )
            == _normalize_answer(
                prediction
            )
        )

    if metrics_config[
        "numerical_accuracy"
    ]:
        numerical_accuracy = _numerical_accuracy(
            reference=reference,
            prediction=prediction,
        )

    return (
        exact_match,
        normalized_exact_match,
        numerical_accuracy,
    )


def _generate_prediction(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    evaluation_config: dict[str, Any],
    device: torch.device,
) -> str:
    """
    Generate one deterministic or configured evaluation prediction.
    """

    if not isinstance(
        prompt,
        str,
    ) or not prompt.strip():
        raise ValueError(
            "Evaluation prompt cannot be empty."
        )

    max_new_tokens = int(
        evaluation_config[
            "max_new_tokens"
        ]
    )

    temperature = float(
        evaluation_config[
            "temperature"
        ]
    )

    do_sample = bool(
        evaluation_config[
            "do_sample"
        ]
    )

    top_p = float(
        evaluation_config[
            "top_p"
        ]
    )

    generation_config = evaluation_config[
        "generation"
    ]

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=(
            tokenizer.model_max_length
            if tokenizer.model_max_length
            and tokenizer.model_max_length < 1_000_000
            else 131072
        ),
    )

    input_ids = encoded[
        "input_ids"
    ].to(
        device
    )

    attention_mask = encoded[
        "attention_mask"
    ].to(
        device
    )

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            "Evaluation generation requires exactly one input sequence."
        )

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": float(
            generation_config[
                "repetition_penalty"
            ]
        ),
        "no_repeat_ngram_size": int(
            generation_config[
                "no_repeat_ngram_size"
            ]
        ),
        "early_stopping": bool(
            generation_config[
                "early_stopping"
            ]
        ),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs[
            "temperature"
        ] = temperature

        generation_kwargs[
            "top_p"
        ] = top_p

    with torch.inference_mode():
        generated = model.generate(
            **generation_kwargs
        )

    if not isinstance(
        generated,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Model.generate() did not return a tensor."
        )

    input_length = input_ids.shape[
        1
    ]

    generated_tokens = generated[
        0,
        input_length:,
    ]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    if not isinstance(
        prediction,
        str,
    ):
        raise RuntimeError(
            "Tokenizer returned a non-string prediction."
        )

    return prediction.strip()


def _prepare_loss_inputs(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Construct a causal-LM sequence containing the prompt followed
    by the target and mask the prompt portion from the loss.

    This avoids using the project's training labels directly because
    the current tokenizer stores prompt and target as separate
    padded sequences.
    """

    if not isinstance(
        prompt,
        str,
    ) or not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    if not isinstance(
        target,
        str,
    ) or not target.strip():
        raise ValueError(
            "Target cannot be empty."
        )

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=(
            tokenizer.model_max_length
            if tokenizer.model_max_length
            and tokenizer.model_max_length < 1_000_000
            else 131072
        ),
    )[
        "input_ids"
    ]

    target_tokens = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=(
            tokenizer.model_max_length
            if tokenizer.model_max_length
            and tokenizer.model_max_length < 1_000_000
            else 131072
        ),
    )[
        "input_ids"
    ]

    if not prompt_tokens:
        raise RuntimeError(
            "Prompt tokenization produced no tokens."
        )

    if not target_tokens:
        raise RuntimeError(
            "Target tokenization produced no tokens."
        )

    combined_tokens = (
        prompt_tokens
        + target_tokens
    )

    input_ids = torch.tensor(
        [combined_tokens],
        dtype=torch.long,
        device=device,
    )

    labels = torch.tensor(
        [
            (
                [-100] * len(prompt_tokens)
                + target_tokens
            )
        ],
        dtype=torch.long,
        device=device,
    )

    return (
        input_ids,
        labels,
    )


def _calculate_example_loss(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str,
    device: torch.device,
) -> float:
    """
    Calculate causal language-model loss only over target tokens.
    """

    (
        input_ids,
        labels,
    ) = _prepare_loss_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        target=target,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    model.eval()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss = getattr(
        outputs,
        "loss",
        None,
    )

    if not isinstance(
        loss,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Model did not return a scalar evaluation loss."
        )

    if loss.ndim != 0:
        raise RuntimeError(
            "Evaluation loss must be scalar."
        )

    if not torch.isfinite(
        loss.detach()
    ):
        raise FloatingPointError(
            "Evaluation loss is non-finite."
        )

    return float(
        loss.detach()
        .cpu()
        .item()
    )


def _safe_perplexity(
    loss: float | None,
) -> float | None:
    """
    Convert mean negative log-likelihood into perplexity.
    """

    if loss is None:
        return None

    if not math.isfinite(
        loss
    ):
        raise FloatingPointError(
            "Mean loss must be finite."
        )

    if loss > 100:
        return float(
            "inf"
        )

    perplexity = math.exp(
        loss
    )

    if not math.isfinite(
        perplexity
    ):
        return float(
            "inf"
        )

    return perplexity


def _evaluate_split(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    samples: list[dict[str, Any]],
    split_name: str,
    evaluation_config: dict[str, Any],
    device: torch.device,
) -> tuple[
    EvaluationMetrics,
    list[EvaluationExample],
]:
    """
    Evaluate one complete validation or test split.
    """

    if split_name not in {
        "validation",
        "test",
    }:
        raise ValueError(
            f"Unsupported evaluation split: {split_name}"
        )

    dataloader = _build_evaluation_dataloader(
        samples=samples,
        batch_size=int(
            evaluation_config[
                "batch_size"
            ]
        ),
    )

    examples: list[
        EvaluationExample
    ] = []

    exact_matches: list[
        bool
    ] = []

    normalized_matches: list[
        bool
    ] = []

    numerical_matches: list[
        bool
    ] = []

    losses: list[
        float
    ] = []

    for batch in dataloader:
        if not isinstance(
            batch,
            dict,
        ):
            raise RuntimeError(
                "Evaluation DataLoader produced an invalid batch."
            )

        prompts = batch.get(
            "prompt"
        )

        targets = batch.get(
            "target"
        )

        identifiers = batch.get(
            "id"
        )

        filenames = batch.get(
            "filename"
        )

        if not isinstance(
            prompts,
            list,
        ) or not isinstance(
            targets,
            list,
        ) or not isinstance(
            identifiers,
            list,
        ) or not isinstance(
            filenames,
            list,
        ):
            raise RuntimeError(
                "Evaluation DataLoader batch fields must be lists."
            )

        if not (
            len(prompts)
            == len(targets)
            == len(identifiers)
            == len(filenames)
            == 1
        ):
            raise RuntimeError(
                "Evaluation generation currently requires "
                "exactly one example per batch."
            )

        prompt = str(
            prompts[0]
        )

        target = str(
            targets[0]
        )

        example_id = str(
            identifiers[0]
        )

        filename = str(
            filenames[0]
        )

        try:
            prediction = _generate_prediction(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                evaluation_config=evaluation_config,
                device=device,
            )

            (
                exact_match,
                normalized_exact_match,
                numerical_accuracy,
            ) = _calculate_example_metrics(
                reference=target,
                prediction=prediction,
                evaluation_config=evaluation_config,
            )

            loss = _calculate_example_loss(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target=target,
                device=device,
            )

            losses.append(
                loss
            )

            if exact_match is not None:
                exact_matches.append(
                    exact_match
                )

            if normalized_exact_match is not None:
                normalized_matches.append(
                    normalized_exact_match
                )

            if numerical_accuracy is not None:
                numerical_matches.append(
                    numerical_accuracy
                )

            examples.append(
                EvaluationExample(
                    example_id=example_id,
                    filename=filename,
                    reference=target,
                    prediction=prediction,
                    exact_match=exact_match,
                    normalized_exact_match=(
                        normalized_exact_match
                    ),
                    numerical_accuracy=(
                        numerical_accuracy
                    ),
                    failed=False,
                    error=None,
                )
            )

        except Exception as error:
            examples.append(
                EvaluationExample(
                    example_id=example_id,
                    filename=filename,
                    reference=target,
                    prediction="",
                    exact_match=None,
                    normalized_exact_match=None,
                    numerical_accuracy=None,
                    failed=True,
                    error=str(
                        error
                    ),
                )
            )

    evaluated_examples = sum(
        not example.failed
        for example in examples
    )

    failed_examples = sum(
        example.failed
        for example in examples
    )

    if evaluated_examples == 0:
        raise RuntimeError(
            f"Every example failed during {split_name} evaluation."
        )

    mean_loss = (
        sum(losses)
        / len(losses)
        if losses
        else None
    )

    metrics = EvaluationMetrics(
        split=split_name,
        total_examples=len(
            examples
        ),
        evaluated_examples=evaluated_examples,
        failed_examples=failed_examples,
        exact_match=(
            sum(exact_matches)
            / len(exact_matches)
            if exact_matches
            else None
        ),
        normalized_exact_match=(
            sum(normalized_matches)
            / len(normalized_matches)
            if normalized_matches
            else None
        ),
        numerical_accuracy=(
            sum(numerical_matches)
            / len(numerical_matches)
            if numerical_matches
            else None
        ),
        mean_loss=mean_loss,
        perplexity=_safe_perplexity(
            mean_loss
        ),
    )

    return (
        metrics,
        examples,
    )


def _save_json(
    path: Path,
    payload: Any,
) -> None:
    """
    Atomically save JSON evaluation artifacts.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        temporary_path.replace(
            path
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def _save_evaluation_artifacts(
    checkpoint_path: Path,
    results: dict[str, EvaluationMetrics],
    examples: dict[str, list[EvaluationExample]],
    evaluation_config: dict[str, Any],
) -> list[Path]:
    """
    Save configured evaluation metrics, predictions, and failures.
    """

    output_config = evaluation_config[
        "output"
    ]

    output_directory = Path(
        output_config[
            "directory"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_files: list[
        Path
    ] = []

    if output_config[
        "save_metrics"
    ]:
        metrics_payload = {
            "checkpoint": str(
                checkpoint_path
            ),
            "metrics": {
                split_name: asdict(
                    metrics
                )
                for split_name, metrics
                in results.items()
            },
        }

        metrics_path = (
            output_directory
            / "metrics.json"
        )

        _save_json(
            path=metrics_path,
            payload=metrics_payload,
        )

        created_files.append(
            metrics_path
        )

    if output_config[
        "save_predictions"
    ]:
        prediction_payload: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for split_name, split_examples in examples.items():
            prediction_payload[
                split_name
            ] = [
                asdict(
                    example
                )
                for example in split_examples
                if not example.failed
            ]

        predictions_path = (
            output_directory
            / "predictions.json"
        )

        _save_json(
            path=predictions_path,
            payload=prediction_payload,
        )

        created_files.append(
            predictions_path
        )

    if output_config[
        "save_failed_examples"
    ]:
        failed_payload: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for split_name, split_examples in examples.items():
            failed_payload[
                split_name
            ] = [
                asdict(
                    example
                )
                for example in split_examples
                if example.failed
            ]

        failed_path = (
            output_directory
            / "failed_examples.json"
        )

        _save_json(
            path=failed_path,
            payload=failed_payload,
        )

        created_files.append(
            failed_path
        )

    return created_files


def evaluate(
    splits: dict[
        str,
        list[dict[str, Any]],
    ] | None = None,
) -> EvaluationResult:
    """
    Run the complete evaluation pipeline.

    When splits is omitted, the evaluator loads the project's
    preprocessed validation and test datasets.

    Returns:
        Structured evaluation results containing metrics,
        predictions, failures, and checkpoint information.
    """

    evaluation_config = _load_evaluation_config()

    _validate_evaluation_config(
        evaluation_config
    )

    if not evaluation_config[
        "enabled"
    ]:
        raise RuntimeError(
            "Evaluation is disabled in the configuration."
        )

    seed = int(
        evaluation_config[
            "reproducibility"
        ][
            "seed"
        ]
    )

    _set_reproducibility_seed(
        seed
    )

    tokenizer = get_tokenizer()

    _validate_tokenizer(
        tokenizer
    )

    if splits is None:
        preprocessed_dataset = (
            get_preprocessed_dataset()
        )

        if not isinstance(
            preprocessed_dataset,
            dict,
        ):
            raise RuntimeError(
                "Preprocessed dataset must be a dictionary."
            )

        splits = preprocessed_dataset

    if not isinstance(
        splits,
        dict,
    ):
        raise TypeError(
            "splits must be a dictionary."
        )

    checkpoint_path = _resolve_checkpoint_path()

    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(
        f"Checkpoint : {checkpoint_path}"
    )
    print(
        f"Seed       : {seed}"
    )

    model = _load_evaluation_model(
        checkpoint_path
    )

    device = _get_model_device(
        model
    )

    print(
        f"Device     : {device}"
    )

    results: dict[
        str,
        EvaluationMetrics,
    ] = {}

    examples: dict[
        str,
        list[EvaluationExample],
    ] = {}

    configured_splits = evaluation_config[
        "splits"
    ]

    for split_name in (
        "validation",
        "test",
    ):
        if not configured_splits[
            split_name
        ]:
            continue

        if split_name not in splits:
            raise RuntimeError(
                f"Configured evaluation split '{split_name}' "
                "does not exist in the preprocessed dataset."
            )

        split_samples = splits[
            split_name
        ]

        if not isinstance(
            split_samples,
            list,
        ):
            raise TypeError(
                f"Evaluation split '{split_name}' must be a list."
            )

        metrics, split_examples = _evaluate_split(
            model=model,
            tokenizer=tokenizer,
            samples=split_samples,
            split_name=split_name,
            evaluation_config=evaluation_config,
            device=device,
        )

        results[
            split_name
        ] = metrics

        examples[
            split_name
        ] = split_examples

        print("=" * 80)
        print(
            f"EVALUATION SPLIT: {split_name.upper()}"
        )
        print("=" * 80)
        print(
            f"Total Examples        : "
            f"{metrics.total_examples:,}"
        )
        print(
            f"Evaluated Examples    : "
            f"{metrics.evaluated_examples:,}"
        )
        print(
            f"Failed Examples       : "
            f"{metrics.failed_examples:,}"
        )

        if metrics.exact_match is not None:
            print(
                f"Exact Match           : "
                f"{metrics.exact_match:.6f}"
            )

        if metrics.normalized_exact_match is not None:
            print(
                f"Normalized Exact Match: "
                f"{metrics.normalized_exact_match:.6f}"
            )

        if metrics.numerical_accuracy is not None:
            print(
                f"Numerical Accuracy    : "
                f"{metrics.numerical_accuracy:.6f}"
            )

        if metrics.mean_loss is not None:
            print(
                f"Mean Loss             : "
                f"{metrics.mean_loss:.6f}"
            )

        if metrics.perplexity is not None:
            print(
                f"Perplexity            : "
                f"{metrics.perplexity:.6f}"
            )

    if not results:
        raise RuntimeError(
            "No evaluation splits are enabled."
        )

    created_files = _save_evaluation_artifacts(
        checkpoint_path=checkpoint_path,
        results=results,
        examples=examples,
        evaluation_config=evaluation_config,
    )

    result: EvaluationResult = {
        "checkpoint": str(
            checkpoint_path
        ),
        "device": str(
            device
        ),
        "seed": seed,
        "metrics": {
            split_name: asdict(
                metrics
            )
            for split_name, metrics
            in results.items()
        },
        "artifacts": [
            str(path)
            for path in created_files
        ],
    }

    return result


def _run_unit_tests() -> None:
    """
    Run evaluator tests that do not require loading the production
    QLoRA model.
    """

    print("=" * 80)
    print("EVALUATOR MODULE TEST")
    print("=" * 80)

    evaluation_config = {
        "enabled": True,
        "batch_size": 1,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "top_p": 1.0,
        "metrics": {
            "exact_match": True,
            "normalized_exact_match": True,
            "numerical_accuracy": True,
        },
        "splits": {
            "validation": True,
            "test": True,
        },
        "generation": {
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "early_stopping": True,
        },
        "output": {
            "directory": "artifacts/evaluation",
            "save_predictions": True,
            "save_metrics": True,
            "save_failed_examples": True,
        },
        "reproducibility": {
            "seed": 42,
        },
    }

    _validate_evaluation_config(
        evaluation_config
    )

    print(
        "Configuration          : PASSED"
    )

    sample_dataset = [
        {
            "prompt": "Question",
            "target": "42",
            "id": "test-1",
            "filename": "test.json",
        }
    ]

    dataset = EvaluationDataset(
        sample_dataset
    )

    if len(dataset) != 1:
        raise RuntimeError(
            "EvaluationDataset length test failed."
        )

    print(
        "Dataset                : PASSED"
    )

    dataloader = _build_evaluation_dataloader(
        samples=sample_dataset,
        batch_size=1,
    )

    batches = list(
        dataloader
    )

    if len(batches) != 1:
        raise RuntimeError(
            "Evaluation DataLoader test failed."
        )

    batch = batches[0]

    if not isinstance(
        batch,
        dict,
    ):
        raise RuntimeError(
            "Evaluation DataLoader returned an invalid batch."
        )

    if batch["prompt"] != [
        "Question"
    ]:
        raise RuntimeError(
            "Evaluation DataLoader prompt collation failed."
        )

    print(
        "DataLoader             : PASSED"
    )

    if _normalize_answer(
        "  42. "
    ) != "42":
        raise RuntimeError(
            "Answer normalization test failed."
        )

    if not _numerical_accuracy(
        reference="42",
        prediction="42.000000",
    ):
        raise RuntimeError(
            "Numerical accuracy positive test failed."
        )

    if _numerical_accuracy(
        reference="42",
        prediction="43",
    ):
        raise RuntimeError(
            "Numerical accuracy negative test failed."
        )

    print(
        "Metric Logic           : PASSED"
    )

    _set_reproducibility_seed(
        42
    )

    first_python = random.random()
    first_numpy = float(
        np.random.random()
    )
    first_torch = float(
        torch.rand(
            1
        ).item()
    )

    _set_reproducibility_seed(
        42
    )

    second_python = random.random()
    second_numpy = float(
        np.random.random()
    )
    second_torch = float(
        torch.rand(
            1
        ).item()
    )

    if first_python != second_python:
        raise RuntimeError(
            "Python evaluation seed is not deterministic."
        )

    if first_numpy != second_numpy:
        raise RuntimeError(
            "NumPy evaluation seed is not deterministic."
        )

    if first_torch != second_torch:
        raise RuntimeError(
            "PyTorch evaluation seed is not deterministic."
        )

    print(
        "Reproducibility        : PASSED"
    )

    print("=" * 80)
    print(
        "Evaluator module is ready for "
        "production evaluation."
    )
    print(
        "Status                  : PASSED"
    )
    print("=" * 80)


def main() -> None:
    """
    Run the evaluator module validation.

    The default module test deliberately does not load the
    production 3B QLoRA model. Full evaluation is invoked through
    evaluate() after training has produced a valid checkpoint.
    """

    _run_unit_tests()


if __name__ == "__main__":
    main()