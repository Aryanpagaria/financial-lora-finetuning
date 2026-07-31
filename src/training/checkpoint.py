import os
from datetime import datetime

import torch

from src.utils.config_loader import load_configs


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
) -> None:
    """
    Save the latest training checkpoint.
    """

    config = load_configs()

    checkpoint_directory = config["checkpoint"]["save_directory"]
    checkpoint_filename = config["checkpoint"]["file_name"]

    os.makedirs(
        checkpoint_directory,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_directory,
        checkpoint_filename,
    )

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "timestamp": datetime.utcnow().isoformat(),
        "training_config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(f"\nLatest checkpoint saved to:\n{checkpoint_path}")




def load_checkpoint(
    model,
    optimizer,
    scheduler,
):
    """
    Load a saved checkpoint.
    """

    config = load_configs()

    resume_config = config["checkpoint"]["resume"]

    if not resume_config["enabled"]:

        print("Starting training from scratch.")

        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "start_epoch": 0,
            "global_step": 0,
            "best_loss": float("inf"),
        }

    checkpoint_path = resume_config["path"]

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    print(f"\nLoaded checkpoint: {checkpoint_path}")

    print_checkpoint_info(
        checkpoint,
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "start_epoch": checkpoint["epoch"]+1,
        "global_step": checkpoint["global_step"],
        "best_loss": checkpoint["best_validation_loss"],
    }

def save_best_checkpoint(
    model,
    validation_loss: float,
) -> None:
    """
    Save the best-performing LoRA adapter.
    """

    config = load_configs()

    best_model_directory = (
        config["checkpoint"]["best_model_directory"]
    )

    os.makedirs(
        best_model_directory,
        exist_ok=True,
    )

    model.save_pretrained(
        best_model_directory,
    )

    print(
        "\nNew best model saved "
        f"(Validation Loss: {validation_loss:.6f})"
    )

    print(
        f"Location: {best_model_directory}"
    )

def export_lora_adapter(
    model,
    tokenizer,
) -> None:
    """
    Export the trained LoRA adapter and tokenizer.
    """

    config = load_configs()

    export_directory = (
        config["checkpoint"]["lora_export_directory"]
    )

    os.makedirs(
        export_directory,
        exist_ok=True,
    )

    model.save_pretrained(
        export_directory,
    )

    tokenizer.save_pretrained(
        export_directory,
    )

    print(
        "\nLoRA adapter exported successfully."
    )

    print(
        f"Location: {export_directory}"
    )

def checkpoint_exists() -> bool:
    """
    Check whether a training checkpoint exists.
    """

    config = load_configs()

    checkpoint_directory = (
        config["checkpoint"]["save_directory"]
    )

    checkpoint_filename = (
        config["checkpoint"]["file_name"]
    )

    checkpoint_path = os.path.join(
        checkpoint_directory,
        checkpoint_filename,
    )

    return os.path.exists(
        checkpoint_path,
    )

def print_checkpoint_info(
    checkpoint: dict,
) -> None:
    """
    Display checkpoint information.
    """

    print("=" * 80)
    print("CHECKPOINT INFORMATION")
    print("=" * 80)

    print(
        f"Epoch                : {checkpoint['epoch']}"
    )

    print(
        f"Global Step          : {checkpoint['global_step']}"
    )

    print(
        "Best Validation Loss : "
        f"{checkpoint['best_validation_loss']:.6f}"
    )

    if "timestamp" in checkpoint:

        print(
            f"Saved At             : {checkpoint['timestamp']}"
        )

    print("=" * 80)

def cleanup_old_checkpoints() -> None:
    """
    Remove old checkpoints while keeping
    only the most recent checkpoint files.
    """

    config = load_configs()

    checkpoint_directory = (
        config["checkpoint"]["save_directory"]
    )

    if not os.path.exists(
        checkpoint_directory,
    ):
        return

    keep_last = (
        config["checkpoint"]["save_total_limit"]
    )

    checkpoint_files = [

        os.path.join(
            checkpoint_directory,
            file_name,
        )

        for file_name in os.listdir(
            checkpoint_directory,
        )

        if file_name.endswith(".pt")
    ]

    checkpoint_files.sort(
        key=os.path.getmtime,
        reverse=True,
    )

    for checkpoint_file in checkpoint_files[keep_last:]:

        os.remove(
            checkpoint_file,
        )

        print(
            f"Removed old checkpoint: "
            f"{checkpoint_file}"
        )