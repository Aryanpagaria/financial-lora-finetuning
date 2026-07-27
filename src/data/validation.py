from typing import Any

from src.data.cleaning import get_clean_dataset

RawDataset = dict[str, list[dict[str, Any]]]


REQUIRED_SAMPLE_FIELDS = {
    "pre_text",
    "post_text",
    "table",
    "qa",
    "filename",
    "id",
}

REQUIRED_QA_FIELDS = {
    "question",
    "answer",
    "exe_ans",
}


def _has_training_target(qa: dict[str, Any]) -> bool:
    """Return True if a QA sample contains a valid training target."""

    return bool(qa["answer"].strip() or qa["exe_ans"].strip())

def _validate_dataset_structure(dataset: RawDataset) -> None:

    required_splits = [
        "train",
        "validation",
        "test",
    ]

    for split in required_splits:

        if split not in dataset:
            raise ValueError(f"Missing dataset split: '{split}'")

        if not isinstance(dataset[split], list):
            raise TypeError(f"Split '{split}' must be a list.")

        if len(dataset[split]) == 0:
            raise ValueError(f"Split '{split}' is empty.")



def _validate_required_fields(dataset: RawDataset) -> None:
    

    for split, records in dataset.items():

        for index, sample in enumerate(records):

            missing_fields = REQUIRED_SAMPLE_FIELDS - sample.keys()

            if missing_fields:
                raise ValueError(
                    f"{split}[{index}] is missing fields: {sorted(missing_fields)}"
                )

            qa = sample["qa"]

            if not isinstance(qa, dict):
                raise TypeError(f"{split}[{index}]['qa'] must be a dictionary.")

            missing_qa_fields = REQUIRED_QA_FIELDS - qa.keys()

            if missing_qa_fields:
                raise ValueError(
                    f"{split}[{index}]['qa'] is missing fields: "
                    f"{sorted(missing_qa_fields)}"
                )



def _validate_field_types(dataset: RawDataset) -> None:
    """Validate the data types of cleaned fields."""

    for split, records in dataset.items():

        for index, sample in enumerate(records):

            if not isinstance(sample["pre_text"], str):
                raise TypeError(
                    f"{split}[{index}]['pre_text'] must be a string."
                )

            if not isinstance(sample["post_text"], str):
                raise TypeError(
                    f"{split}[{index}]['post_text'] must be a string."
                )

            if not isinstance(sample["table"], str):
                raise TypeError(
                    f"{split}[{index}]['table'] must be a string."
                )

            qa = sample["qa"]

            if not isinstance(qa["question"], str):
                raise TypeError(
                    f"{split}[{index}]['qa']['question'] must be a string."
                )

            if not isinstance(qa["answer"], str):
                raise TypeError(
                    f"{split}[{index}]['qa']['answer'] must be a string."
                )

            if not isinstance(qa["exe_ans"], str):
                raise TypeError(
                    f"{split}[{index}]['qa']['exe_ans'] must be a string."
                )



def _validate_field_values(dataset: RawDataset) -> None:
    """Validate that required text fields are not empty."""

    for split, records in dataset.items():

        for index, sample in enumerate(records):

            qa = sample["qa"]

            if not sample["pre_text"].strip() and not sample["post_text"].strip():
                raise ValueError(
                    f"{split}[{index}] has no textual context."
                )

            if not sample["table"].strip():
                raise ValueError(
                    f"{split}[{index}] has an empty table."
                )

            if not qa["question"].strip():
                raise ValueError(
                    f"{split}[{index}] has an empty question."
                )

            # ✅ Only check that at least one target exists
            if not _has_training_target(qa):
                raise ValueError(
                    f"{split}[{index}] has no valid training target."
                )


def _validate_unique_ids(dataset: RawDataset) -> None:
    """Ensure sample IDs are unique within each split."""

    for split, records in dataset.items():

        seen_ids: set[str] = set()

        for index, sample in enumerate(records):

            sample_id = sample["id"]

            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate ID '{sample_id}' found in {split}[{index}]."
                )

            seen_ids.add(sample_id)

def _print_validation_summary(dataset: RawDataset) -> None:
    """Print a summary after successful validation."""

    total_samples = sum(len(records) for records in dataset.values())

    print("=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    for split, records in dataset.items():
        print(f"{split:<12}: {len(records)} samples")

    print("-" * 80)
    print(f"Total Samples : {total_samples}")
    print("Validation    : PASSED ✓")


def get_validated_dataset(verbose: bool = True,) -> RawDataset:
    """Return a validated cleaned dataset."""

    dataset = get_clean_dataset()
    _validate_dataset_structure(dataset)
    _validate_required_fields(dataset)
    _validate_field_types(dataset)
    _validate_field_values(dataset)
    _validate_unique_ids(dataset)

    if verbose:
        _print_validation_summary(dataset)

    return dataset


if __name__ == "__main__":
    get_validated_dataset()

