from pathlib import Path
import torch
from typing import Any
from transformers import AutoTokenizer
from src.utils.config_loader import load_configs
from src.utils.log import get_logger
import torch
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,

)



def load_inference_config() -> dict[str, Any]:
    """
    Load and validate the inference configuration.
    """

    config = load_configs()

    checkpoint_config = config["checkpoint"]

    lora_adapter_directory = Path(
        checkpoint_config["lora_export_directory"]
    )

    if not lora_adapter_directory.exists():

        raise FileNotFoundError(
            "LoRA adapter directory not found:\n"
            f"{lora_adapter_directory}"
        )

    adapter_weights = (
        lora_adapter_directory
        / "adapter_model.safetensors"
    )

    if not adapter_weights.exists():

        raise FileNotFoundError(
            "adapter_model.safetensors not found:\n"
            f"{adapter_weights}"
        )

    tokenizer_file = (
        lora_adapter_directory
        / "tokenizer.json"
    )

    if not tokenizer_file.exists():

        raise FileNotFoundError(
            "tokenizer.json not found:\n"
            f"{tokenizer_file}"
        )

    print("=" * 80)
    print("INFERENCE CONFIGURATION")
    print("=" * 80)
    print(
        f"Base Model : "
        f"{config['model']['name']}"
    )
    print(
        f"LoRA Path  : "
        f"{lora_adapter_directory}"
    )
    print("=" * 80)

    return config





def load_tokenizer(
    config: dict[str, Any],
):
    """
    Load the tokenizer exported with the LoRA adapter.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        config["checkpoint"][
            "lora_export_directory"
        ],
        trust_remote_code=True,
    )

    tokenizer.padding_side = (
        config["tokenizer"]["padding_side"]
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print("=" * 80)
    print("TOKENIZER LOADED")
    print("=" * 80)
    print(f"Vocabulary Size : {len(tokenizer)}")
    print(f"Padding Side    : {tokenizer.padding_side}")
    print("=" * 80)

    return tokenizer




def load_model(
    config: dict[str, Any],
):
    """
    Load the base language model for inference.
    """

    quantization = (
        config["quantization"]
    )

    compute_dtype = getattr(
        torch,
        quantization[
            "bnb_4bit_compute_dtype"
        ],
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=quantization[
                "load_in_4bit"
            ],
            bnb_4bit_quant_type=quantization[
                "bnb_4bit_quant_type"
            ],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quantization[
                "bnb_4bit_use_double_quant"
            ],
        )
    )

    print("=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config[
            "model"
        ]["name"],
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.eval()
    model.config.use_cache = True

    for parameter in model.parameters():

        parameter.requires_grad = False

    print("=" * 80)
    print("MODEL READY FOR INFERENCE")
    print("=" * 80)
    print(
        f"Device : "
        f"{next(model.parameters()).device}"
    )
    print("=" * 80)

    return model




def load_lora_adapter(
    model: Any,
    config: dict[str, Any],
) -> Any:
    """
    Load the trained LoRA adapter.
    """

    adapter_directory = Path(
        config["checkpoint"][
            "lora_export_directory"
        ]
    )

    print("=" * 80)
    print("LOADING LORA ADAPTER")
    print("=" * 80)

    model = PeftModel.from_pretrained(
        model=model,
        model_id=str(
            adapter_directory,
        ),
        is_trainable=False,
    )

    model.eval()

    print(
        f"Adapter : {adapter_directory}"
    )

    print("=" * 80)

    return model





def interactive_chat(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> None:
    """
    Interactive CLI chat.
    """

    logger = get_logger(
        "inference",
    )

    history = []

    exit_commands = (
        config["chat"][
            "exit_commands"
        ]
    )

    print("=" * 80)
    print("Financial LoRA Assistant")
    print("Type 'exit' to quit.")
    print("=" * 80)

    while True:

        user_prompt = input(
            "\nYou : "
        ).strip()

        if (
            user_prompt.lower()
            in exit_commands
        ):

            print(
                "\nGoodbye!"
            )

            break

        history = build_messages(
            history,
            user_prompt,
        )

        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
        )

        response = result[
            "response"
        ]

        history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        max_history = config["chat"][
            "max_history"
        ]

        if len(history) > max_history:

            history = [
                history[0],
                *history[-(max_history - 1):]
            ]

        logger.info(
            f"User: {user_prompt}"
        )

        logger.info(
            f"Assistant: {response}"
        )

        print("\nAssistant:\n")

        print(response)

        print()

        print("-" * 80)
        print("Generation Statistics")
        print("-" * 80)

        print(
            f"Prompt Tokens    : "
            f"{result['prompt_tokens']}"
        )

        print(
            f"Generated Tokens : "
            f"{result['generated_tokens']}"
        )

        print(
            f"Generation Time  : "
            f"{result['generation_time']:.2f} sec"
        )

        print(
            f"Tokens / Second  : "
            f"{result['tokens_per_second']:.2f}"
        )

        print(
            f"Temperature      : "
            f"{result['temperature']}"
        )

        print(
            f"Top-p            : "
            f"{result['top_p']}"
        )

        print("-" * 80)


def build_messages(
    history: list[dict[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    """
    Build the complete conversation history.
    """

    if not history:

        history.append(
            {
                "role": "system",
                "content":
                (
                    "You are a helpful financial "
                    "assistant."
                ),
            }
        )

    history.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return history









def main() -> None:

    config = load_inference_config()

    tokenizer = load_tokenizer(
        config,
    )

    model = load_model(
        config,
    )

    model = load_lora_adapter(
        model,
        config,
    )

    interactive_chat(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


if __name__ == "__main__":

    main()