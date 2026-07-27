from pathlib import Path
import json
import pprint
import requests


# Configuration


DATA_DIR = Path("data/raw/finqa")

FILES = {
    "train.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}

SPLIT_MAPPING = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}



# Internal Helper Functions


def _download_file(url: str, destination: Path) -> None:
    """
    Download a file from a URL.

    Args:
        url: Source URL.
        destination: Local file path.

    Raises:
        requests.HTTPError:
            If the download fails.
    """

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    
    
    print(response.status_code)
    print(response.headers.get("content-length"))
    print(response.url)



# Data Acquisition


def ensure_raw_data() -> None:
    """
    Download any missing raw dataset files.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():

        destination = DATA_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[SKIP] {filename}")
            continue
        print(f"[DOWNLOAD] {filename}")

        _download_file(url, destination)

    print("\nRaw dataset is ready.\n")


def load_raw_files() -> dict[str, list]:
    """
    Load all dataset splits from disk.

    Returns:
        Dictionary containing train, validation and test splits.
    """

    raw_data = {}

    for split, filename in SPLIT_MAPPING.items():

        path = DATA_DIR / filename

        print(f"Loading: {path}")

        with open(path, "r", encoding="utf-8") as file:
            raw_data[split] = json.load(file)

    return raw_data



# Validation


def verify_raw_dataset(raw_data: dict[str, list]) -> None:
    """
    Verify the integrity of the raw dataset.
    """

    required_splits = [
        "train",
        "validation",
        "test",
    ]

    for split in required_splits:

        if split not in raw_data:
            raise ValueError(f"Missing dataset split: {split}")

        if not isinstance(raw_data[split], list):
            raise TypeError(f"{split} must be a list.")

        if len(raw_data[split]) == 0:
            raise ValueError(f"{split} split is empty.")



# Developer Utilities


def inspect_dataset(raw_data: dict[str, list]) -> None:
    """
    Inspect the structure of the raw dataset.

    This function is intended only for development and debugging.
    """

    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)

    print("\nDataset Statistics")
    print("-" * 40)

    for split, records in raw_data.items():
        print(f"{split:<12}: {len(records)} samples")

    sample = raw_data["train"][0]

    print("\nFirst Training Sample")
    print("-" * 40)
    pprint.pprint(sample, width=120)

    print("\nTop-Level Keys")
    print("-" * 40)
    print(list(sample.keys()))

    print("\nField Types")
    print("-" * 40)

    for key, value in sample.items():
        print(f"{key:<20} -> {type(value).__name__}")



# Public API


def get_raw_dataset() -> dict[str, list]:
    """
    Download, load and validate the raw FinQA dataset.

    Returns:
        Raw dataset dictionary.
    """

    ensure_raw_data()

    raw_data = load_raw_files()

    verify_raw_dataset(raw_data)

    return raw_data


                                                                             
# Local Testing


if __name__ == "__main__":

    raw_data = get_raw_dataset()

    inspect_dataset(raw_data)