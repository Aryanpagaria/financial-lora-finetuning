from typing import Any
from transformers import AutoTokenizer
from src.data.validation import get_validated_dataset

RawDataset = dict[str, list[dict[str, Any]]]
ProcessedDataset = dict[str, list[dict[str, Any]]]
_TOKENIZER = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
)


def _truncate_text(
    text: str,
    max_tokens: int,
) -> str:
    """
    Truncate text using tokenizer
    token count instead of characters.
    """

    if not text.strip():

        return ""

    token_ids = _TOKENIZER.encode(

        text,

        add_special_tokens=False,

    )

    token_ids = token_ids[
        :max_tokens
    ]

    return _TOKENIZER.decode(

        token_ids,

        skip_special_tokens=True,

    )



def _build_context(
    sample: dict[str, Any],
) -> str:
    """
    Build a compact financial context
    that fits within the training
    token budget.
    """

    sections: list[str] = []

    background = _truncate_text(

        sample["pre_text"],

        150,

    )

    table = _truncate_text(

        sample["table"],

        180,

    )

    post_text = _truncate_text(

        sample["post_text"],

        40,

    )

    if background:

        sections.append(

            "Background:\n"

            f"{background}"

        )

    if table:

        sections.append(

            "Financial Table:\n"

            f"{table}"

        )

    if post_text:

        sections.append(

            "Additional Context:\n"

            f"{post_text}"

        )

    return "\n\n".join(
        sections
    )

def _extract_question(
    sample: dict[str, Any],
) -> str:
    """
    Return the financial question.
    """

    return (
        sample["qa"]["question"]
        .strip()
    )


def _extract_answer(
    sample: dict[str, Any],
) -> str:
    """
    Return the preferred answer.
    """

    qa = sample["qa"]

    if qa["answer"].strip():

        return qa["answer"].strip()

    return qa["exe_ans"].strip()


def _build_system_prompt() -> str:
    """
    System instruction used for
    every training conversation.
    """

    return (
        "You are a highly accurate financial AI assistant.\n\n"
        "Answer financial questions using ONLY the provided "
        "financial context.\n\n"
        "If the context does not contain enough information, "
        "say that the answer cannot be determined from the "
        "provided information.\n\n"
        "Do not fabricate facts."
    )


def _build_user_prompt(
    context: str,
    question: str,
) -> str:
    """
    Build the user message containing
    the financial context and question.
    """

    return (
        "Financial Context:\n\n"
        f"{context}\n\n"
        "Question:\n\n"
        f"{question}\n\n"
        "Answer the question using only the provided financial context."
    )


def _build_messages(
    context: str,
    question: str,
    answer: str,
) -> list[dict[str, str]]:
    """
    Build a Qwen-compatible
    conversation.
    """

    return [

        {
            "role": "system",
            "content": _build_system_prompt(),
        },

        {
            "role": "user",
            "content": _build_user_prompt(
                context,
                question,
            ),
        },

        {
            "role": "assistant",
            "content": answer,
        },
    ]


def _preprocess_sample(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one raw sample into
    a chat-format training example.
    """

    context = _build_context(
        sample,
    )

    question = _extract_question(
        sample,
    )

    answer = _extract_answer(
        sample,
    )

    messages = _build_messages(
        context,
        question,
        answer,
    )

    return {

        "id": sample["id"],

        "filename": sample["filename"],

        "context": context,

        "question": question,

        "answer": answer,

        "messages": messages,

    }


def _preprocess_split(
    split: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Preprocess every sample in one split.
    """

    processed_split = []

    for sample in split:

        processed_split.append(
            _preprocess_sample(
                sample,
            )
        )

    return processed_split


def _print_preprocessing_summary(
    dataset: ProcessedDataset,
) -> None:
    """
    Print dataset preprocessing summary.
    """

    total_samples = sum(
        len(split)
        for split in dataset.values()
    )

    print("=" * 80)
    print("PREPROCESSING REPORT")
    print("=" * 80)

    for split_name, split in dataset.items():

        print(
            f"{split_name:<12}: "
            f"{len(split):,} samples"
        )

    print("-" * 80)

    print(
        f"Total Samples : {total_samples:,}"
    )

    print("Format        : Chat Conversations")

    print("Status        : SUCCESS")

    print("=" * 80)


def get_preprocessed_dataset() -> ProcessedDataset:
    """
    Return the complete
    chat-formatted dataset.
    """

    raw_dataset = get_validated_dataset()

    processed_dataset: ProcessedDataset = {}

    for split_name, split in raw_dataset.items():

        processed_dataset[
            split_name
        ] = _preprocess_split(
            split,
        )

    _print_preprocessing_summary(
        processed_dataset,
    )

    return processed_dataset


if __name__ == "__main__":

    dataset = get_preprocessed_dataset()

    sample = dataset["train"][0]

    print("=" * 80)
    print("TRAINING CONVERSATION")
    print("=" * 80)

    for message in sample["messages"]:

        print(
            f"\n[{message['role'].upper()}]"
        )

        print(
            message["content"]
        )

    print()

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        sample["answer"]
    )