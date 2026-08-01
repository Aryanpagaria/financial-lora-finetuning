from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_configs() -> dict[str, Any]:
    """
    Load and merge every project configuration file.
    """

    configuration = {}

    configuration.update(
        _load_yaml(
            CONFIG_DIR / "model" / "model.yaml"
        )
    )

    configuration.update(
        _load_yaml(
            CONFIG_DIR / "training" / "training.yaml"
        )
    )

    configuration.update(
        _load_yaml(
            CONFIG_DIR / "checkpoint" / "checkpoint.yaml"
        )
    )

    configuration.update(
        _load_yaml(
            CONFIG_DIR / "logging" / "logging.yaml"
        )
    )

    configuration.update(
        _load_yaml(
            CONFIG_DIR / "evaluation" / "evaluation.yaml"
        )
    )

    configuration.update(
        _load_yaml(
            CONFIG_DIR / "data" / "data.yaml"
        )
    )

    return configuration


if __name__ == "__main__":

    config = load_configs()

    print("=" * 80)
    print("CONFIG LOADER")
    print("=" * 80)

    for key, value in config.items():
        print(f"{key}:")
        print(value)
        print()