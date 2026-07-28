from typing import Any

from src.data.validation import get_validated_dataset

RawDataset = dict[str, list[dict[str, Any]]]
ProcessedDataset = dict[str, list[dict[str, str]]]


def _build_context(sample: dict[str, Any]) -> str:
    

    sections: list[str] = []
 
    if sample["pre_text"].strip():
        sections.append(
            f"Background:\n{sample['pre_text']}"
        )

    if sample["table"].strip():
        sections.append(
            f"Financial Table:\n{sample['table']}"
        )

    if sample["post_text"].strip():
        sections.append(
            f"Additional Context:\n{sample['post_text']}"
        )

    return "\n\n".join(sections)


def _select_target(sample: dict[str, Any]) -> str:
    

    qa = sample["qa"]

    if qa["answer"].strip():
        return qa["answer"]

    return qa["exe_ans"]


def _create_prompt(context: str, question: str) -> str:
    

    return (
        "### Instruction\n"
        "Answer the financial question using the provided financial context.\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{question}\n\n"
        "### Response\n"

    )


def _preprocess_sample(sample: dict[str, Any]) -> dict[str, str]:

    context = _build_context(sample)

    question = sample["qa"]["question"]

    target = _select_target(sample)

    prompt = _create_prompt(context, question)

    return {
        "id": sample["id"],
        "filename": sample["filename"],
        "prompt": prompt,
        "target": target,
        "answer": sample["qa"]["answer"],
        "exe_ans": sample["qa"]["exe_ans"],
    }



def _preprocess_split(split: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Preprocess all samples in a dataset split."""

    processed_split: list[dict[str, str]] = []

    for sample in split:
        processed_sample = _preprocess_sample(sample)
        processed_split.append(processed_sample)

    return processed_split


def _print_preprocessing_summary(dataset: ProcessedDataset) -> None:
    

    total_samples = sum(len(split) for split in dataset.values())

    print("=" * 80)
    print("PREPROCESSING REPORT")
    print("=" * 80)

    for split_name, samples in dataset.items():
        print(f"{split_name:<12}: {len(samples)} samples")

    print("-" * 80)
    print(f"Total Samples : {total_samples}")
    print("Status        : COMPLETED ✓")


def get_preprocessed_dataset() -> ProcessedDataset:
    """Return the fully preprocessed dataset."""

    dataset = get_validated_dataset()

    processed_dataset: ProcessedDataset = {}

    for split_name, split in dataset.items():
        processed_dataset[split_name] = _preprocess_split(split)

    _print_preprocessing_summary(processed_dataset)

    return processed_dataset


if __name__ == "__main__":

    processed_dataset = get_preprocessed_dataset()

    sample = processed_dataset["train"][0]

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(sample["prompt"])

    print("\n" + "=" * 80)
    print("TARGET")
    print("=" * 80)
    print(sample["target"])