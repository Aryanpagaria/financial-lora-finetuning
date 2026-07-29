from typing import Any

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import AutoModelForCausalLM

from src.utils.config_loader import load_configs


def _load_model() -> Any:
    config = load_configs()

    model_name = config["model"]["name"]

    print("Before from_pretrained()")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )

    print("After from_pretrained()")

    return model    


def _prepare_model(model: Any) -> Any:
    """Prepare the model for parameter-efficient fine-tuning."""

    model = prepare_model_for_kbit_training(model)

    return model


def _create_lora_config() -> LoraConfig:
    """Create the LoRA configuration."""

    config = load_configs()

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    return lora_config


def _apply_lora(
    model: Any,
    lora_config: LoraConfig,
) -> Any:
    """Apply LoRA adapters to the model."""

    model = get_peft_model(
        model=model,
        peft_config=lora_config,
    )

    return model


def print_trainable_parameters(model: Any) -> None:
    """Print the number of trainable parameters."""

    trainable_params = 0
    all_params = 0

    for parameter in model.parameters():
        all_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()

    percentage = 100 * trainable_params / all_params

    print("=" * 80)
    print("TRAINABLE PARAMETERS")
    print("=" * 80)

    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Total Parameters     : {all_params:,}")
    print(f"Trainable Percentage : {percentage:.4f}%")


def get_model() -> Any:
    """Return the LoRA-enabled language model."""

    print("Loading base model...")
    model = _load_model()

    print("Preparing model...")
    model = _prepare_model(model)

    print("Creating LoRA config...")
    lora_config = _create_lora_config()

    print("Applying LoRA...")
    model = _apply_lora(
        model,
        lora_config,
    )

    print("Printing trainable parameters...")
    print_trainable_parameters(model)

    print("Returning model...")
    return model


if __name__ == "__main__":
    model = get_model()