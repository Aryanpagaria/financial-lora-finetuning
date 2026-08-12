from typing import Any

import torch

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from src.utils.config_loader import load_configs


ModelConfig = dict[str, Any]


def _load_model_config() -> ModelConfig:
    """
    Load and validate the complete project configuration.

    Returns:
        The complete merged project configuration.
    """

    config = load_configs()

    if not isinstance(
        config,
        dict,
    ):
        raise RuntimeError(
            "Loaded project configuration must be a dictionary."
        )

    if "model" not in config:
        raise KeyError(
            "Missing 'model' configuration."
        )

    if "lora" not in config:
        raise KeyError(
            "Missing 'lora' configuration."
        )

    model_config = config["model"]

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError(
            "'model' configuration must be a dictionary."
        )

    if "quantization" not in model_config:
        raise KeyError(
            "Missing 'model.quantization' configuration."
        )

    quantization_config = model_config[
        "quantization"
    ]

    if not isinstance(
        quantization_config,
        dict,
    ):
        raise TypeError(
            "'model.quantization' configuration must be a dictionary."
        )

    required_model_keys = {
        "name",
        "quantization",
        "device",
        "gradient_checkpointing",
        "use_cache",
    }

    missing_model_keys = sorted(
        required_model_keys
        - set(model_config.keys())
    )

    if missing_model_keys:
        raise KeyError(
            "Missing model configuration keys: "
            f"{missing_model_keys}"
        )

    required_quantization_keys = {
        "enabled",
        "load_in_4bit",
        "quantization_type",
        "use_double_quantization",
        "compute_dtype",
    }

    missing_quantization_keys = sorted(
        required_quantization_keys
        - set(quantization_config.keys())
    )

    if missing_quantization_keys:
        raise KeyError(
            "Missing model.quantization configuration keys: "
            f"{missing_quantization_keys}"
        )

    lora_config = config["lora"]

    if not isinstance(
        lora_config,
        dict,
    ):
        raise TypeError(
            "'lora' configuration must be a dictionary."
        )

    return config


def _build_quantization_config(
    config: ModelConfig,
) -> BitsAndBytesConfig | None:
    """
    Build the BitsAndBytes configuration from the canonical
    model.quantization YAML schema.
    """

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "config must be a dictionary."
        )

    model_config = config.get(
        "model"
    )

    if not isinstance(
        model_config,
        dict,
    ):
        raise RuntimeError(
            "Missing or invalid 'model' configuration."
        )

    quantization = model_config.get(
        "quantization"
    )

    if not isinstance(
        quantization,
        dict,
    ):
        raise RuntimeError(
            "Missing or invalid 'model.quantization' configuration."
        )

    enabled = quantization[
        "enabled"
    ]

    if not isinstance(
        enabled,
        bool,
    ):
        raise TypeError(
            "model.quantization.enabled must be boolean."
        )

    if not enabled:
        return None

    load_in_4bit = quantization[
        "load_in_4bit"
    ]

    if not isinstance(
        load_in_4bit,
        bool,
    ):
        raise TypeError(
            "model.quantization.load_in_4bit must be boolean."
        )

    if not load_in_4bit:
        return None

    quantization_type = quantization[
        "quantization_type"
    ]

    if quantization_type not in {
        "nf4",
        "fp4",
    }:
        raise ValueError(
            "model.quantization.quantization_type must be "
            "'nf4' or 'fp4'."
        )

    use_double_quantization = quantization[
        "use_double_quantization"
    ]

    if not isinstance(
        use_double_quantization,
        bool,
    ):
        raise TypeError(
            "model.quantization.use_double_quantization "
            "must be boolean."
        )

    compute_dtype_name = quantization[
        "compute_dtype"
    ]

    if not isinstance(
        compute_dtype_name,
        str,
    ):
        raise TypeError(
            "model.quantization.compute_dtype must be a string."
        )

    supported_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if compute_dtype_name not in supported_dtypes:
        raise ValueError(
            "Unsupported model.quantization.compute_dtype: "
            f"{compute_dtype_name}. Supported values: "
            f"{sorted(supported_dtypes)}"
        )

    compute_dtype = supported_dtypes[
        compute_dtype_name
    ]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit quantization requires CUDA, "
            "but CUDA is not available."
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization_type,
        bnb_4bit_use_double_quant=use_double_quantization,
        bnb_4bit_compute_dtype=compute_dtype,
    )

def _build_lora_config(
    config: ModelConfig,
) -> LoraConfig:
    """
    Build the LoRA configuration used
    for parameter-efficient fine-tuning.
    """

    lora = config["lora"]

    required_keys = {
        "r",
        "alpha",
        "dropout",
        "bias",
        "target_modules",
    }

    missing_keys = (
        required_keys
        - lora.keys()
    )

    if missing_keys:
        raise KeyError(
            "Missing LoRA configuration "
            f"keys: {sorted(missing_keys)}"
        )

    rank = int(lora["r"])

    if rank <= 0:
        raise ValueError(
            "LoRA rank must be greater than zero."
        )

    alpha = float(
        lora["alpha"]
    )

    if alpha <= 0:
        raise ValueError(
            "LoRA alpha must be greater than zero."
        )

    dropout = float(
        lora["dropout"]
    )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "LoRA dropout must be in "
            "the range [0, 1)."
        )

    target_modules = lora[
        "target_modules"
    ]

    if not isinstance(
        target_modules,
        list,
    ) or not target_modules:
        raise ValueError(
            "LoRA target_modules must be "
            "a non-empty list."
        )

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=str(
            lora["bias"]
        ),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def _validate_runtime(
    config: ModelConfig,
) -> None:
    """
    Validate the runtime environment required
    for the configured QLoRA training setup.
    """

    quantization = config["model"]["quantization"]

    load_in_4bit = bool(
        quantization["load_in_4bit"]
    )

    if load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit QLoRA training requires a "
            "CUDA-enabled GPU in the current "
            "training configuration."
        )

    if load_in_4bit:
        cuda_device_count = torch.cuda.device_count()

        if cuda_device_count < 1:
            raise RuntimeError(
                "CUDA was reported as unavailable "
                "despite 4-bit quantization being enabled."
            )

        print("=" * 80)
        print("RUNTIME VALIDATION")
        print("=" * 80)
        print(
            f"CUDA Devices : {cuda_device_count}"
        )
        print(
            f"CUDA Device  : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"CUDA Version : "
            f"{torch.version.cuda}"
        )
        print("=" * 80)



def _load_base_model(
    config: ModelConfig,
    quantization_config: BitsAndBytesConfig,
) -> PreTrainedModel:
    """
    Load the base causal language model.

    The model is loaded without LoRA adapters.
    LoRA is applied in a separate step so that
    model construction remains explicit and testable.
    """

    model_name = config["model"]["name"]

    if not isinstance(
        model_name,
        str,
    ) or not model_name.strip():
        raise ValueError(
            "Model name must be a non-empty string."
        )

    print("=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)
    print(
        f"Model : {model_name}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if model is None:
        raise RuntimeError(
            "Base model loading returned None."
        )

    print(
        "Base model loaded successfully."
    )

    print("=" * 80)

    return model




def _configure_base_model(
    model: PreTrainedModel,
) -> PreTrainedModel:
    """
    Configure the base model for causal-language-model
    fine-tuning before LoRA adapters are attached.
    """

    if not hasattr(
        model,
        "config",
    ):
        raise RuntimeError(
            "Loaded model does not expose a configuration."
        )

    model.config.use_cache = False

    if hasattr(
        model.config,
        "pretraining_tp",
    ):
        model.config.pretraining_tp = 1

    if (
        hasattr(
            model.config,
            "pad_token_id",
        )
        and model.config.pad_token_id is None
    ):
        if model.config.eos_token_id is None:
            raise RuntimeError(
                "Neither pad_token_id nor "
                "eos_token_id is available."
            )

        model.config.pad_token_id = (
            model.config.eos_token_id
        )

    return model



def _prepare_for_qlora(
    model: PreTrainedModel,
    config: ModelConfig,
) -> PreTrainedModel:
    """
    Prepare a quantized model for QLoRA training.
    """

    quantization = config[
        "quantization"
    ]

    if not bool(
        quantization["load_in_4bit"]
    ):
        return model

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    return model



def _enable_training_optimizations(
    model: PreTrainedModel,
) -> PreTrainedModel:
    """
    Enable memory-saving training optimizations.
    """

    model.config.use_cache = False

    if not model.is_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

    model.enable_input_require_grads()

    return model

def _attach_lora_adapter(
    model: PreTrainedModel,
    lora_config: LoraConfig,
) -> PreTrainedModel:
    """
    Attach the trainable LoRA adapters to the
    prepared base model.
    """

    model = get_peft_model(
        model,
        lora_config,
    )

    if not isinstance(
        model,
        PeftModel,
    ):
        raise RuntimeError(
            "LoRA adapter attachment failed. "
            "Expected a PeftModel."
        )

    return model




def _freeze_base_parameters(
    model: PreTrainedModel,
) -> None:
    """
    Ensure base-model parameters are frozen.

    Only LoRA adapter parameters should remain
    trainable for this QLoRA configuration.
    """

    for name, parameter in model.named_parameters():

        if "lora_" in name.lower():
            continue

        parameter.requires_grad = False


def _get_parameter_statistics(
    model: PreTrainedModel,
) -> tuple[int, int, float]:
    """
    Calculate trainable and total parameter counts.
    """

    trainable_parameters = 0
    total_parameters = 0

    for parameter in model.parameters():

        parameter_count = parameter.numel()

        total_parameters += parameter_count

        if parameter.requires_grad:
            trainable_parameters += parameter_count

    if total_parameters == 0:
        raise RuntimeError(
            "Model contains zero parameters."
        )

    percentage = (
        trainable_parameters
        / total_parameters
        * 100.0
    )

    return (
        trainable_parameters,
        total_parameters,
        percentage,
    )



def _validate_trainable_parameters(
    model: PreTrainedModel,
) -> None:
    """
    Verify that only LoRA parameters are trainable
    and report the actual number of trainable scalar
    parameters.
    """

    trainable_parameter_names: list[str] = []

    trainable_parameter_count = 0

    total_parameter_count = 0

    for name, parameter in model.named_parameters():

        parameter_count = parameter.numel()

        total_parameter_count += parameter_count

        if parameter.requires_grad:

            trainable_parameter_names.append(
                name
            )

            trainable_parameter_count += (
                parameter_count
            )

    if total_parameter_count == 0:

        raise RuntimeError(
            "Model contains zero parameters."
        )

    if trainable_parameter_count == 0:

        raise RuntimeError(
            "No trainable parameters were found."
        )

    non_lora_trainable = [

        name

        for name in trainable_parameter_names

        if "lora_" not in name.lower()

    ]

    if non_lora_trainable:

        raise RuntimeError(
            "Non-LoRA parameters are trainable: "
            f"{non_lora_trainable[:10]}"
        )

    trainable_percentage = (
        trainable_parameter_count
        / total_parameter_count
        * 100.0
    )

    print("=" * 80)
    print("TRAINABLE PARAMETER VALIDATION")
    print("=" * 80)

    print(
        f"Trainable Parameters : "
        f"{trainable_parameter_count:,}"
    )

    print(
        f"Total Parameters     : "
        f"{total_parameter_count:,}"
    )

    print(
        f"Trainable Percentage : "
        f"{trainable_percentage:.4f}%"
    )

    print(
        f"Trainable Tensors    : "
        f"{len(trainable_parameter_names):,}"
    )

    print(
        "Status               : "
        "LoRA parameters only"
    )

    print("=" * 80)

def _print_model_summary(
    model: PreTrainedModel,
    config: ModelConfig,
) -> None:
    """
    Display the final model configuration.
    """

    trainable, total, percentage = (
        _get_parameter_statistics(
            model
        )
    )

    quantization = config[
        "quantization"
    ]

    lora = config[
        "lora"
    ]

    print("=" * 80)
    print("FINAL MODEL SUMMARY")
    print("=" * 80)

    print(
        f"Base Model          : "
        f"{config['model']['name']}"
    )

    print(
        f"Quantization        : "
        f"{quantization['bnb_4bit_quant_type']}"
    )

    print(
        f"4-bit Loading       : "
        f"{quantization['load_in_4bit']}"
    )

    print(
        f"Compute Dtype       : "
        f"{quantization['bnb_4bit_compute_dtype']}"
    )

    print(
        f"LoRA Rank           : "
        f"{lora['r']}"
    )

    print(
        f"LoRA Alpha          : "
        f"{lora['alpha']}"
    )

    print(
        f"LoRA Dropout        : "
        f"{lora['dropout']}"
    )

    print(
        f"Trainable Params    : "
        f"{trainable:,}"
    )

    print(
        f"Total Params        : "
        f"{total:,}"
    )

    print(
        f"Trainable Ratio     : "
        f"{percentage:.4f}%"
    )

    print(
        f"Gradient Checkpoint : "
        f"{model.is_gradient_checkpointing}"
    )

    print(
        f"Use Cache           : "
        f"{model.config.use_cache}"
    )

    print("=" * 80)



def _print_model_memory(
    model: PreTrainedModel,
) -> None:
    """
    Display the model memory footprint when
    the loaded model provides the required API.
    """

    if not hasattr(
        model,
        "get_memory_footprint",
    ):
        print(
            "Model memory footprint is unavailable."
        )
        return

    memory_bytes = model.get_memory_footprint()

    if memory_bytes <= 0:
        raise RuntimeError(
            "Model reported an invalid memory footprint."
        )

    memory_gb = (
        memory_bytes / (1024 ** 3)
    )

    print("=" * 80)
    print("MODEL MEMORY")
    print("=" * 80)
    print(
        f"Memory Footprint : "
        f"{memory_gb:.2f} GB"
    )
    print("=" * 80)


def _sanity_check_model(
    model: PreTrainedModel,
) -> None:
    """
    Perform final validation of the QLoRA model
    before returning it to the training pipeline.
    """

    if not isinstance(
        model,
        PeftModel,
    ):
        raise RuntimeError(
            "Final model is not a PEFT model."
        )

    if not hasattr(
        model,
        "peft_config",
    ):
        raise RuntimeError(
            "Final model does not contain "
            "a PEFT configuration."
        )

    if not model.is_gradient_checkpointing:

        raise RuntimeError(
            "Gradient checkpointing is not enabled."
        )

    if model.config.use_cache:

        raise RuntimeError(
            "use_cache must be False during training."
        )

    trainable_parameter_count = 0

    non_lora_trainable: list[str] = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        trainable_parameter_count += (
            parameter.numel()
        )

        if "lora_" not in name.lower():

            non_lora_trainable.append(
                name
            )

    if trainable_parameter_count == 0:

        raise RuntimeError(
            "Final model contains no trainable "
            "parameters."
        )

    if non_lora_trainable:

        raise RuntimeError(
            "Final model contains trainable "
            "non-LoRA parameters: "
            f"{non_lora_trainable[:10]}"
        )

    if (
        model.config.pad_token_id is None
        and model.config.eos_token_id is None
    ):

        raise RuntimeError(
            "Model has neither pad_token_id "
            "nor eos_token_id."
        )

    if not model.training:

        model.train()

    print("=" * 80)
    print("MODEL SANITY CHECK")
    print("=" * 80)

    print(
        "PEFT Model          : PASSED"
    )

    print(
        "LoRA Parameters     : PASSED"
    )

    print(
        "Base Model Frozen   : PASSED"
    )

    print(
        "Gradient Checkpoint : PASSED"
    )

    print(
        "use_cache=False     : PASSED"
    )

    print(
        "Token Configuration : PASSED"
    )

    print(
        "Training Mode       : PASSED"
    )

    print(
        f"Trainable Parameters: "
        f"{trainable_parameter_count:,}"
    )

    print(
        "Status              : "
        "READY FOR TRAINING"
    )

    print("=" * 80)

def get_model() -> PreTrainedModel:
    """
    Build and validate the complete QLoRA model.

    Returns:
        A PEFT-enabled causal language model
        ready for the training pipeline.
    """

    config = _load_model_config()

    _validate_runtime(
        config,
    )

    quantization_config = (
        _build_quantization_config(
            config,
        )
    )

    lora_config = (
        _build_lora_config(
            config,
        )
    )

    model = _load_base_model(
        config,
        quantization_config,
    )

    model = _configure_base_model(
        model,
    )

    model = _prepare_for_qlora(
        model,
        config,
    )

    model = _enable_training_optimizations(
        model,
    )

    model = _attach_lora_adapter(
        model,
        lora_config,
    )

    _freeze_base_parameters(
        model,
    )

    _validate_trainable_parameters(
        model,
    )

    _sanity_check_model(
        model,
    )

    _print_model_summary(
        model,
        config,
    )

    _print_model_memory(
        model,
    )

    return model


def main() -> None:
    """
    Build the QLoRA model and run all model-level
    validation checks.
    """

    model = get_model()

    print("=" * 80)
    print("MODEL BUILD COMPLETE")
    print("=" * 80)

    print(
        f"Model Type : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Training Mode : "
        f"{model.training}"
    )

    print(
        f"Gradient Checkpointing : "
        f"{model.is_gradient_checkpointing}"
    )

    print(
        f"Use Cache : "
        f"{model.config.use_cache}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()


