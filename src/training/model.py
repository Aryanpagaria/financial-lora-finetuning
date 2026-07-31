from typing import Any

import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.utils.config_loader import load_configs


def _load_quantization_config() -> BitsAndBytesConfig:
    """
    Create the BitsAndBytes quantization configuration.
    """

    config = load_configs()

    quantization = config["quantization"]

    compute_dtype = getattr(
        torch,
        quantization["bnb_4bit_compute_dtype"],
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quantization[
            "bnb_4bit_use_double_quant"
        ],
    )

    return quantization_config


def _load_model(
    quantization_config: BitsAndBytesConfig,
) -> Any:
    """
    Load the base language model.

    If a CUDA-enabled GPU is available, load the model
    using 4-bit quantization for QLoRA training.

    Otherwise, load the model on the CPU for
    development and testing.
    """

    config = load_configs()

    model_name = config["model"]["name"]

    if torch.cuda.is_available():

        print("=" * 80)
        print("LOADING MODEL (GPU MODE)")
        print("=" * 80)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

    else:

        print("=" * 80)
        print("LOADING MODEL (CPU MODE)")
        print("=" * 80)
        print(
            "CUDA is not available. "
            "Loading the model without 4-bit quantization."
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    return model

def _prepare_model(model: Any) -> Any:
    """
    Prepare the model for PEFT training.
    """

    model = prepare_model_for_kbit_training(model)

    return model


def _create_lora_config() -> LoraConfig:
    """
    Create the LoRA configuration.
    """

    config = load_configs()

    lora = config["lora"]

    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora["target_modules"],
    )

    return lora_config


def _apply_peft(
    model: Any,
    lora_config: LoraConfig,
) -> Any:
    """
    Apply LoRA adapters.
    """

    model = get_peft_model(
        model=model,
        peft_config=lora_config,
    )

    return model


def print_trainable_parameters(
    model: Any,
) -> None:
    """
    Print trainable parameter statistics.
    """

    trainable_params = 0
    total_params = 0

    for parameter in model.parameters():

        total_params += parameter.numel()

        if parameter.requires_grad:

            trainable_params += parameter.numel()

    percentage = (
        100 * trainable_params / total_params
    )

    print("=" * 80)
    print("TRAINABLE PARAMETERS")
    print("=" * 80)
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Percentage : {percentage:.4f}%")
    print("=" * 80)


def get_model() -> Any:
    """
    Build and return a PEFT-enabled language model
    ready for QLoRA fine-tuning.
    """

    quantization_config = _load_quantization_config()

    model = _load_model(
        quantization_config,
    )

    model = _prepare_model(
        model,
    )

    model = configure_model(
        model,
    )

    lora_config = _create_lora_config()

    model = _apply_peft(
        model,
        lora_config,
    )

    model = enable_training_optimizations(
        model,
    )

    sanity_check_model(
        model,
    )

    print_model_summary(
        model,
    )

    print_trainable_parameters(
        model,
    )

    return model

def print_model_summary(
    model: Any,
) -> None:
    """
    Display a summary of the loaded model and
    training configuration.
    """

    config = load_configs()

    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Base Model      : {config['model']['name']}")
    print(f"Quantization    : 4-bit")
    print(f"Quant Type      : {config['quantization']['bnb_4bit_quant_type']}")
    print(f"Compute Dtype   : {config['quantization']['bnb_4bit_compute_dtype']}")
    print(f"LoRA Rank       : {config['lora']['r']}")
    print(f"LoRA Alpha      : {config['lora']['alpha']}")
    print(f"LoRA Dropout    : {config['lora']['dropout']}")
    print(f"Device          : {next(model.parameters()).device}")
    print("=" * 80)


def configure_model(
    model: Any,
) -> Any:
    """
    Configure the model before training.
    """

    # Disable cache during training.
    model.config.use_cache = False

    # Set padding token if available.
    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = model.config.eos_token_id

    # Ensure tensor parallelism does not interfere.
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    return model







def sanity_check_model(
    model: Any,
) -> None:
    """
    Verify that the model is correctly configured
    before training begins.
    """

    # Check LoRA adapters.
    if not hasattr(model, "peft_config"):
        raise RuntimeError(
            "LoRA adapters were not attached to the model."
        )

    # Check trainable parameters.
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters == 0:
        raise RuntimeError(
            "No trainable parameters found."
        )

    # Check cache configuration.
    if model.config.use_cache:
        raise RuntimeError(
            "use_cache must be False during training."
        )

    print("=" * 80)
    print("MODEL SANITY CHECK PASSED")
    print("=" * 80)


def enable_training_optimizations(
    model: Any,
) -> Any:
    """
    Enable training optimizations for efficient
    LoRA fine-tuning.
    """

    # Disable KV cache during training.
    model.config.use_cache = False

    # Enable gradient checkpointing to reduce
    # GPU memory usage.
    model.gradient_checkpointing_enable()

    # Required for gradient checkpointing with
    # some transformer architectures.
    model.enable_input_require_grads()

    return model





if __name__ == "__main__":

    model = get_model()