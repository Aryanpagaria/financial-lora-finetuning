from typing import Any
from src.data.preprocessing import get_preprocessed_dataset
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizer

CONFIG_PATH = "configs/model/model.yaml"


def _load_model_config() -> dict[str, Any]:
    """Load model configuration from YAML."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)


    return config


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

    config = _load_model_config()

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

    config = _load_model_config()

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