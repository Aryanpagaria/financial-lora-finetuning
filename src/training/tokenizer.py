from typing import Any
from src.data.preprocessing import get_preprocessed_dataset
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.utils.config_loader import load_configs
import os 


def _configure_tokenizer(config: dict[str, Any]) -> PreTrainedTokenizer:
    """Load and configure the tokenizer."""

    model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = config["tokenizer"]["padding_side"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_tokenizer() -> PreTrainedTokenizer:
    """Return a configured tokenizer."""

    config = load_configs()

    tokenizer = _configure_tokenizer(config)

    return tokenizer



def _tokenize_sample(
    sample: dict[str, str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize a single training sample."""

    prompt_encoding = tokenizer(
        sample["prompt"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    target_encoding = tokenizer(
        sample["target"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    return {
        "input_ids": prompt_encoding["input_ids"],
        "attention_mask": prompt_encoding["attention_mask"],
        "labels": target_encoding["input_ids"],
        "id": sample["id"],
        "filename": sample["filename"],
    }


def _tokenize_split(
    split: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize all samples in a dataset split."""

    tokenized_split = []

    for sample in split:
        tokenized_split.append(
            _tokenize_sample(
                sample,
                tokenizer,
                max_length,
            )
        )

    return tokenized_split




def get_tokenized_dataset() -> dict[str, list[dict[str, Any]]]:
    """Return the tokenized dataset."""

    config = load_configs()

    tokenizer = get_tokenizer()

    dataset = get_preprocessed_dataset()

    max_length = config["training"]["max_seq_length"]

    tokenized_dataset = {}

    for split_name, split in dataset.items():

        tokenized_dataset[split_name] = _tokenize_split(
            split,
            tokenizer,
            max_length,
        )

    return tokenized_dataset

if __name__ == "__main__":

    dataset = get_tokenized_dataset()

    sample = dataset["train"][0]

    print("=" * 80)
    print("TOKENIZED SAMPLE")
    print("=" * 80)

    print("Input IDs:")
    print(sample["input_ids"][:20])

    print()

    print("Attention Mask:")
    print(sample["attention_mask"][:20])

    print()

    print("Labels:")
    print(sample["labels"][:20])


def print_tokenizer_info(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Print tokenizer configuration.
    """

    print("=" * 80)
    print("TOKENIZER INFORMATION")
    print("=" * 80)

    print(
        f"Tokenizer Class : {tokenizer.__class__.__name__}"
    )

    print(
        f"Vocabulary Size : {len(tokenizer)}"
    )

    print(
        f"Padding Side    : {tokenizer.padding_side}"
    )

    print(
        f"Pad Token       : {tokenizer.pad_token}"
    )

    print(
        f"EOS Token       : {tokenizer.eos_token}"
    )

    print(
        f"BOS Token       : {tokenizer.bos_token}"
    )

    print(
        f"UNK Token       : {tokenizer.unk_token}"
    )

    print(
        f"Model Max Length: {tokenizer.model_max_length}"
    )

    print("=" * 80)

def sanity_check_tokenizer(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Validate tokenizer configuration before
    training.
    """

    if tokenizer.pad_token is None:

        raise RuntimeError(
            "Pad token is missing."
        )

    if tokenizer.eos_token is None:

        raise RuntimeError(
            "EOS token is missing."
        )

    if len(tokenizer) == 0:

        raise RuntimeError(
            "Tokenizer vocabulary is empty."
        )

    print("=" * 80)
    print("TOKENIZER SANITY CHECK PASSED")
    print("=" * 80)

def save_tokenizer(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Save tokenizer for inference.
    """

    config = load_configs()

    save_directory = (
        config["checkpoint"][
            "lora_export_directory"
        ]
    )

    os.makedirs(
        save_directory,
        exist_ok=True,
    )

    tokenizer.save_pretrained(
        save_directory,
    )

    print(
        f"\nTokenizer saved to:\n{save_directory}"
    )

def load_saved_tokenizer(
) -> PreTrainedTokenizer:
    """
    Load exported tokenizer.
    """

    config = load_configs()

    save_directory = (
        config["checkpoint"][
            "lora_export_directory"
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        save_directory,
    )

    return tokenizer


def get_tokenizer() -> PreTrainedTokenizer:
    """
    Return a configured tokenizer.
    """

    config = load_configs()

    tokenizer = _configure_tokenizer(
        config,
    )

    sanity_check_tokenizer(
        tokenizer,
    )

    print_tokenizer_info(
        tokenizer,
    )

    return tokenizer