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
    """Load all project configuration files."""

    model_config = _load_yaml(
        CONFIG_DIR / "model" / "model.yaml"
    )

    training_config = _load_yaml(
        CONFIG_DIR / "training" / "training.yaml"
    )

    config = {}

    config.update(model_config)
    config.update(training_config)

    return config


if __name__ == "__main__":

    config = load_configs()

    print("=" * 80)
    print("CONFIG LOADER")
    print("=" * 80)

    for key, value in config.items():
        print(f"{key}:")
        print(value)
        print()