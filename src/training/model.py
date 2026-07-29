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
    """

    config = load_configs()

    model_name = config["model"]["name"]

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantization_config=quantization_config,
        device_map="auto",
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
    Return a PEFT-enabled language model.
    """

    quantization_config = (
        _load_quantization_config()
    )

    model = _load_model(
        quantization_config,
    )

    model = _prepare_model(model)

    lora_config = _create_lora_config()

    model = _apply_peft(
        model,
        lora_config,
    )

    print_trainable_parameters(model)

    return model


if __name__ == "__main__":

    model = get_model()