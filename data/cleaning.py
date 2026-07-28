

from copy import deepcopy
from typing import Any
from src.data.ingestion import get_raw_dataset


RawDataset = dict[str, list[dict[str, Any]]]


def normalize_answer_types(raw_data: RawDataset) -> RawDataset:
    """Convert every ``qa.exe_ans`` value to a string.

    FinQA stores executable answers as a mix of numbers and strings. A
    consistent string representation is required before conversion to an
    Apache Arrow-backed Hugging Face dataset.

    Args:
        raw_data: Raw FinQA records grouped by split.

    Returns:
        A deep-copied dataset with string executable answers.
    """
    cleaned_data: RawDataset = {}

    for split, records in raw_data.items():
        
        cleaned_records: list[dict[str, Any]] = []

        for record in records:
            sample = deepcopy(record)
            sample["qa"]["exe_ans"] = str(sample["qa"]["exe_ans"])
            cleaned_records.append(sample)

        cleaned_data[split] = cleaned_records

    return cleaned_data


def normalize_text_fields(cleaned_data: RawDataset) -> RawDataset:
    """Join pre- and post-table paragraph lists without losing boundaries.

    Paragraphs are separated by two newlines so downstream prompt construction
    retains the original context structure.

    Args:
        cleaned_data: FinQA records grouped by split.

    Returns:
        A deep-copied dataset with ``pre_text`` and ``post_text`` as strings.
    """
    normalized_data: RawDataset = {}

    for split, records in cleaned_data.items():
        normalized_records: list[dict[str, Any]] = []

        for record in records:
            sample = deepcopy(record)
            sample["pre_text"] = "\n\n".join(sample["pre_text"])
            sample["post_text"] = "\n\n".join(sample["post_text"])
            normalized_records.append(sample)

        normalized_data[split] = normalized_records

    return normalized_data


def _format_markdown_cell(cell: str) -> str:
    
    return cell.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _table_to_markdown(table: list[list[str]]) -> str:
    
    header, *rows = table
    separator = ["---"] * len(header)
    markdown_rows = [header, separator, *rows]

    return "\n".join(
        f"| {' | '.join(_format_markdown_cell(cell) for cell in row)} |"
        for row in markdown_rows
    )


def normalize_table_fields(cleaned_data: RawDataset) -> RawDataset:
   
    normalized_data: RawDataset = {}

    for split, records in cleaned_data.items():
        normalized_records: list[dict[str, Any]] = []

        for record in records:
            sample = deepcopy(record)
            sample["table"] = _table_to_markdown(sample["table"])
            normalized_records.append(sample)

        normalized_data[split] = normalized_records

    return normalized_data


def _has_content(value: Any) -> bool:
    """Return whether a scalar or collection contains meaningful content."""
    if isinstance(value, str):
        return bool(value.strip())

    return bool(value)


def _is_valid_sample(sample: dict[str, Any]) -> bool:
    """Check whether a FinQA sample has the minimum training inputs."""
    qa = sample.get("qa")
    if not isinstance(qa, dict):
        return False

    has_target = (
    _has_content(qa.get("answer"))
    or
    _has_content(qa.get("exe_ans"))
    )
    has_context = _has_content(sample.get("pre_text")) or _has_content(
        sample.get("post_text")
    )

    return (
        _has_content(qa.get("question"))
        and has_target
        and _has_content(sample.get("table"))
        and has_context
    )


def remove_invalid_samples(cleaned_data: RawDataset) -> RawDataset:
    
    filtered_data: RawDataset = {}

    for split, records in cleaned_data.items():
        filtered_data[split] = [
            deepcopy(record) for record in records if _is_valid_sample(record)
        ]

    return filtered_data


def get_clean_dataset() -> RawDataset:
    raw_data = get_raw_dataset()

    cleaned = remove_invalid_samples(raw_data)
    cleaned = normalize_answer_types(cleaned)
    cleaned = normalize_text_fields(cleaned)
    cleaned = normalize_table_fields(cleaned)

    return cleaned





if __name__ == "__main__":

    raw_data = get_raw_dataset()

    cleaned_data = normalize_answer_types(raw_data)
    cleaned_data = normalize_text_fields(cleaned_data)
    cleaned_data = normalize_table_fields(cleaned_data)
    cleaned_data = remove_invalid_samples(cleaned_data)

    sample = cleaned_data["train"][0]

    print("=" * 80)
    print("CLEANING DONE")
    print("=" * 80)
    print()
    print("exe_ans type :", type(sample["qa"]["exe_ans"]))
    print("pre_text type:", type(sample["pre_text"]))
    print("post_text type:", type(sample["post_text"]))
    print("table type   :", type(sample["table"]))
    print()
    print(sample["table"][:500])