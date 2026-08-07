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



import time


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a response and return inference statistics.
    """

    generation = config[
        "generation"
    ]

    device = next(
        model.parameters()
    ).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    prompt_tokens = inputs[
        "input_ids"
    ].shape[-1]

    start_time = time.perf_counter()

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=generation[
                "max_new_tokens"
            ],

            temperature=generation[
                "temperature"
            ],

            top_p=generation[
                "top_p"
            ],

            top_k=generation[
                "top_k"
            ],

            do_sample=generation[
                "do_sample"
            ],

            repetition_penalty=generation[
                "repetition_penalty"
            ],

            num_beams=generation[
                "num_beams"
            ],

            use_cache=generation[
                "use_cache"
            ],

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

    end_time = time.perf_counter()

    generated_tokens = outputs[
        0
    ][
        prompt_tokens:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    generated_token_count = (
        generated_tokens.shape[-1]
    )

    generation_time = (
        end_time - start_time
    )

    tokens_per_second = (
        generated_token_count
        / generation_time
        if generation_time > 0
        else 0.0
    )

    return {

        "response": response,

        "prompt_tokens": prompt_tokens,

        "generated_tokens": generated_token_count,

        "generation_time": generation_time,

        "tokens_per_second": tokens_per_second,

        "temperature": generation[
            "temperature"
        ],

        "top_p": generation[
            "top_p"
        ],
    }

def test_inference(
    question: str,
) -> None:
    """
    Run a single inference for notebook testing.
    """

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

    history = []

    history = build_messages(
        history,
        question,
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

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print()

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["response"])

    print()

    print("=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(f"Prompt Tokens    : {result['prompt_tokens']}")
    print(f"Generated Tokens : {result['generated_tokens']}")
    print(f"Generation Time  : {result['generation_time']:.2f} sec")
    print(f"Tokens / Second  : {result['tokens_per_second']:.2f}")
    print(f"Temperature      : {result['temperature']}")
    print(f"Top-p            : {result['top_p']}")

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