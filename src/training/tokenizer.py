from typing import Any
from pathlib import Path
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from src.data.preprocessing import (
    get_preprocessed_dataset,
)

from src.utils.config_loader import (
    load_configs,
)
TokenizedDataset = dict[
    str,
    list[dict[str, Any]],
]


def _configure_tokenizer(
    config: dict[str, Any],
) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(

        config["model"]["name"],

        trust_remote_code=True,

    )

    tokenizer.padding_side = (
        config["tokenizer"][
            "padding_side"
        ]
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return tokenizer


def sanity_check_tokenizer(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Validate tokenizer configuration.
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


def print_tokenizer_info(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Display tokenizer information.
    """

    print("=" * 80)
    print("TOKENIZER INFORMATION")
    print("=" * 80)

    print(
        f"Tokenizer : "
        f"{tokenizer.__class__.__name__}"
    )

    print(
        f"Vocabulary : "
        f"{len(tokenizer):,}"
    )

    print(
        f"Padding Side : "
        f"{tokenizer.padding_side}"
    )

    print(
        f"Pad Token : "
        f"{tokenizer.pad_token}"
    )

    print(
        f"EOS Token : "
        f"{tokenizer.eos_token}"
    )

    print(
        f"Model Max Length : "
        f"{tokenizer.model_max_length}"
    )

    print("=" * 80)


def get_tokenizer(
) -> PreTrainedTokenizer:
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








def _tokenize_split(
    split: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> list[dict[str, Any]]:
    """
    Tokenize an entire dataset split.
    """

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

def _validate_tokenized_supervision(
    tokenized_dataset: TokenizedDataset,
) -> None:
    """
    Validate that every tokenized sample contains at least
    one supervised target token.

    Causal language-model labels may use -100 for ignored
    positions, but a sample containing only -100 labels has
    no training signal and can produce an invalid mean loss.
    """

    if not isinstance(
        tokenized_dataset,
        dict,
    ):
        raise TypeError(
            "tokenized_dataset must be a dictionary."
        )

    for split_name, samples in tokenized_dataset.items():

        if not isinstance(
            samples,
            list,
        ):
            raise TypeError(
                f"Tokenized split '{split_name}' must be a list."
            )

        invalid_samples: list[int] = []

        for index, sample in enumerate(
            samples
        ):

            if not isinstance(
                sample,
                dict,
            ):
                raise TypeError(
                    f"Tokenized sample {index} in split "
                    f"'{split_name}' must be a dictionary."
                )

            if "labels" not in sample:
                raise RuntimeError(
                    f"Tokenized sample {index} in split "
                    f"'{split_name}' is missing labels."
                )

            labels = sample["labels"]

            if not isinstance(
                labels,
                list,
            ):
                raise TypeError(
                    f"Labels for sample {index} in split "
                    f"'{split_name}' must be a list."
                )

            valid_label_count = sum(
                1
                for label in labels
                if label != -100
            )

            if valid_label_count == 0:
                invalid_samples.append(
                    index
                )

        if invalid_samples:

            preview = invalid_samples[:20]

            raise ValueError(
                f"Split '{split_name}' contains "
                f"{len(invalid_samples)} samples with no "
                f"valid supervised labels. "
                f"All labels are -100. "
                f"First invalid indices: {preview}"
            )

def get_tokenized_dataset(
) -> TokenizedDataset:
    """
    Return the complete
    tokenized dataset.
    """

    config = load_configs()

    tokenizer = get_tokenizer()

    dataset = get_preprocessed_dataset()

    max_length = (
        config["training"][
            "max_seq_length"
        ]
    )

    tokenized_dataset: TokenizedDataset = {}

    for split_name, split in dataset.items():

        tokenized_dataset[
            split_name
        ] = _tokenize_split(

            split,

            tokenizer,

            max_length,

        )

    print_dataset_statistics(
        tokenized_dataset,
    )
    
    _validate_tokenized_supervision(
        tokenized_dataset
    )

    return tokenized_dataset

def save_tokenizer(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Save tokenizer for inference.
    """

    config = load_configs()

    save_directory = Path(

        config["checkpoint"][
            "lora_export_directory"
        ]

    )

    save_directory.mkdir(

        parents=True,

        exist_ok=True,

    )

    tokenizer.save_pretrained(
        save_directory,
    )

    print("=" * 80)
    print("TOKENIZER SAVED")
    print("=" * 80)
    print(save_directory)
    print("=" * 80)




def load_saved_tokenizer(
) -> PreTrainedTokenizer:
    """
    Load exported tokenizer.
    """

    config = load_configs()

    tokenizer = AutoTokenizer.from_pretrained(

        config["checkpoint"][
            "lora_export_directory"
        ],

        trust_remote_code=True,

    )

    return tokenizer


def print_dataset_statistics(
    dataset: TokenizedDataset,
) -> None:
    """
    Print tokenized dataset statistics.
    """

    print("=" * 80)
    print("TOKENIZED DATASET")
    print("=" * 80)

    total = 0

    for split_name, split in dataset.items():

        print(
            f"{split_name:<12}: "
            f"{len(split):,} samples"
        )

        total += len(split)

    print("-" * 80)

    print(
        f"Total Samples : {total:,}"
    )

    print("=" * 80)



def _build_prompt_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Return only the system and user
    messages.

    These form the prompt presented
    to the language model.
    """

    prompt_messages: list[
        dict[str, str]
    ] = []

    for message in messages:

        if message["role"] == "assistant":

            break

        prompt_messages.append(
            message
        )

    return prompt_messages


def _build_answer_message(
    messages: list[dict[str, str]],
) -> str:
    """
    Return the assistant response
    used as the training target.
    """

    for message in messages:

        if message["role"] == "assistant":

            return message[
                "content"
            ]

    raise RuntimeError(
        "Assistant response missing."
    )



def _tokenize_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt_messages: list[
        dict[str, str]
    ],
) -> list[int]:
    """
    Tokenize the prompt
    (system + user).
    """

    prompt_text = tokenizer.apply_chat_template(

        conversation=prompt_messages,

        tokenize=False,

        add_generation_prompt=True,

    )

    prompt_ids = tokenizer(

        prompt_text,

        add_special_tokens=False,

    )[
        "input_ids"
    ]

    return prompt_ids


def _tokenize_answer(
    tokenizer: PreTrainedTokenizer,
    answer: str,
) -> list[int]:
    """
    Tokenize the assistant answer.
    """

    answer_ids = tokenizer(

        answer,

        add_special_tokens=False,

    )[
        "input_ids"
    ]

    answer_ids.append(
        tokenizer.eos_token_id
    )

    return answer_ids




def _tokenize_sample(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, Any]:
    """
    Tokenize one training conversation.
    """

    prompt_messages = (
        _build_prompt_messages(
            sample["messages"]
        )
    )

    answer = (
        _build_answer_message(
            sample["messages"]
        )
    )

    prompt_ids = (
        _tokenize_prompt(
            tokenizer,
            prompt_messages,
        )
    )

    answer_ids = (
        _tokenize_answer(
            tokenizer,
            answer,
        )
    )

    input_ids = (
        prompt_ids
        + answer_ids
    )

    attention_mask = (
        [1]
        * len(input_ids)
    )

    labels = (
        [-100]
        * len(prompt_ids)
        + answer_ids
    )

    input_ids = input_ids[
        :max_length
    ]

    attention_mask = attention_mask[
        :max_length
    ]

    labels = labels[
        :max_length
    ]

    padding = (
        max_length
        - len(input_ids)
    )

    if padding > 0:

        input_ids.extend(

            [
                tokenizer.pad_token_id
            ]
            * padding

        )

        attention_mask.extend(

            [0]
            * padding

        )

        labels.extend(

            [-100]
            * padding

        )

    return {

        "id": sample["id"],

        "filename": sample["filename"],

        "messages": sample["messages"],

        "input_ids": input_ids,

        "attention_mask": attention_mask,

        "labels": labels,

    }

def preview_training_labels(
    tokenizer: PreTrainedTokenizer,
    labels: list[int],
) -> None:
    """
    Display only the tokens that
    contribute to the training loss.
    """

    answer_tokens = [

        token

        for token in labels

        if token != -100

    ]

    print("=" * 80)
    print("TRAINING LABELS")
    print("=" * 80)

    print(

        tokenizer.decode(

            answer_tokens,

            skip_special_tokens=False,

        )

    )

    print("=" * 80)




if __name__ == "__main__":

    tokenizer = get_tokenizer()

    tokenized_dataset = (
        get_tokenized_dataset()
    )

    save_tokenizer(
        tokenizer,
    )

    sample = tokenized_dataset[
        "train"
    ][0]

    print("=" * 80)
    print("TOKENIZED SAMPLE")
    print("=" * 80)

    print(
        f"Input Length      : {len(sample['input_ids'])}"
    )

    print(
        f"Attention Length  : {len(sample['attention_mask'])}"
    )

    print(
        f"Label Length      : {len(sample['labels'])}"
    )

    print()

    print("=" * 80)
    print("FULL CONVERSATION")
    print("=" * 80)

    print(

        tokenizer.decode(

            sample["input_ids"],

            skip_special_tokens=False,

        )

    )

    print()

    preview_training_labels(

        tokenizer,

        sample["labels"],

    )


    print()

    prompt_messages = _build_prompt_messages(
        sample["messages"],
    )

    prompt_ids = _tokenize_prompt(
        tokenizer,
        prompt_messages,
    )

    answer = _build_answer_message(
        sample["messages"],
    )

    answer_ids = _tokenize_answer(
        tokenizer,
        answer,
    )

    print("=" * 80)
    print("TOKEN STATISTICS")
    print("=" * 80)

    print(
        f"Prompt Tokens    : {len(prompt_ids)}"
    )

    print(
        f"Answer Tokens    : {len(answer_ids)}"
    )

    print(
        f"Total Tokens     : {len(prompt_ids) + len(answer_ids)}"
    )

    print(
        f"Maximum Length   : 512"
    )

    print("=" * 80)
    print()
    print("=" * 80)
    print("PROMPT TOKENS")
    print("=" * 80)

    prompt_messages = _build_prompt_messages(
        sample["messages"],
    )

    prompt_ids = _tokenize_prompt(
        tokenizer,
        prompt_messages,
    )

    print(len(prompt_ids))

    print()

    answer = _build_answer_message(
        sample["messages"],
    )

    answer_ids = _tokenize_answer(
        tokenizer,
        answer,
    )

    print("=" * 80)
    print("ANSWER TOKENS")
    print("=" * 80)

    print(len(answer_ids))

    print()

    print("=" * 80)
    print("TOTAL TOKENS")
    print("=" * 80)
    

    print(len(prompt_ids) + len(answer_ids))