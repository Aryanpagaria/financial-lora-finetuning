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
) -> str:
    """
    Save a versioned checkpoint and update latest.pt.
    """

    config = load_configs()

    checkpoint_directory = (
        config["checkpoint"]["save_directory"]
    )

    os.makedirs(
        checkpoint_directory,
        exist_ok=True,
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

    versioned_checkpoint = os.path.join(
        checkpoint_directory,
        f"epoch_{epoch}.pt",
    )

    latest_checkpoint = os.path.join(
        checkpoint_directory,
        "latest.pt",
    )

    torch.save(
        checkpoint,
        versioned_checkpoint,
    )

    torch.save(
        checkpoint,
        latest_checkpoint,
    )

    if not verify_checkpoint(
        latest_checkpoint,
    ):

        raise RuntimeError(
            "Checkpoint verification failed."
        )

    print(
        "\nCheckpoint saved successfully."
    )

    print(
        f"Latest   : {latest_checkpoint}"
    )

    print(
        f"Version  : {versioned_checkpoint}"
    )

    return latest_checkpoint

def load_checkpoint(
    model,
    optimizer,
    scheduler,
) -> dict:
    """
    Load a training checkpoint if resume is enabled.
    """

    config = load_configs()

    resume_config = config["checkpoint"]["resume"]

    if not resume_config["enabled"]:

        print("=" * 80)
        print("TRAINING MODE")
        print("=" * 80)
        print("Starting training from scratch.")
        print("=" * 80)

        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "start_epoch": 0,
            "global_step": 0,
            "best_loss": float("inf"),
        }

    checkpoint_path = resume_config["path"]

    if not os.path.isfile(
        checkpoint_path,
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print("=" * 80)
    print("LOADING CHECKPOINT")
    print("=" * 80)
    print(
        f"Checkpoint : {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    required_keys = [
        "epoch",
        "global_step",
        "best_validation_loss",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ]

    missing_keys = [

        key

        for key in required_keys

        if key not in checkpoint

    ]

    if missing_keys:

        raise RuntimeError(
            "Checkpoint is corrupted.\n"
            f"Missing keys: {missing_keys}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = (
        checkpoint["epoch"]
    )

    global_step = (
        checkpoint["global_step"]
    )

    best_loss = (
        checkpoint["best_validation_loss"]
    )

    print_checkpoint_info(
        checkpoint,
    )

    print("=" * 80)
    print("RESUME INFORMATION")
    print("=" * 80)
    print(
        f"Next Epoch          : {start_epoch + 1}"
    )
    print(
        f"Global Step         : {global_step}"
    )
    print(
        f"Best Validation Loss: {best_loss:.6f}"
    )
    print("=" * 80)

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "start_epoch": start_epoch,
        "global_step": global_step,
        "best_loss": best_loss,
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
    Remove old versioned checkpoints while
    keeping the newest checkpoint files.
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

        if (
            file_name.endswith(".pt")
            and file_name != "latest.pt"
        )
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

def verify_checkpoint(
    checkpoint_path: str,
) -> bool:
    """
    Verify that a checkpoint can be loaded.
    """

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        required_keys = {
            "epoch",
            "global_step",
            "best_validation_loss",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }

        missing_keys = (
            required_keys
            - checkpoint.keys()
        )

        if missing_keys:

            raise RuntimeError(
                "Checkpoint missing keys: "
                f"{missing_keys}"
            )

        print(
            "Checkpoint verification passed."
        )

        return True

    except Exception as error:

        print(
            "Checkpoint verification failed."
        )

        print(error)

        return False


def get_latest_checkpoint_path() -> str:
    """
    Return the newest checkpoint path.
    """

    config = load_configs()

    checkpoint_directory = (
        config["checkpoint"]["save_directory"]
    )

    if not os.path.exists(
        checkpoint_directory,
    ):

        raise FileNotFoundError(
            "Checkpoint directory not found."
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

    if not checkpoint_files:

        raise FileNotFoundError(
            "No checkpoints found."
        )

    checkpoint_files.sort(
        key=os.path.getmtime,
        reverse=True,
    )

    return checkpoint_files[0]



def delete_checkpoint(
    checkpoint_path: str,
) -> None:
    """
    Delete a checkpoint file.
    """

    if not os.path.exists(
        checkpoint_path,
    ):

        print(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

        return

    os.remove(
        checkpoint_path,
    )

    print(
        f"Deleted checkpoint:\n"
        f"{checkpoint_path}"
    )


def list_checkpoints() -> list[str]:
    """
    Return all available checkpoints.
    """

    config = load_configs()

    checkpoint_directory = (
        config["checkpoint"]["save_directory"]
    )

    if not os.path.exists(
        checkpoint_directory,
    ):

        return []

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

    return checkpoint_files



def print_available_checkpoints() -> None:
    """
    Print all saved checkpoints.
    """

    checkpoint_files = list_checkpoints()

    print("=" * 80)
    print("AVAILABLE CHECKPOINTS")
    print("=" * 80)

    if not checkpoint_files:

        print(
            "No checkpoints found."
        )

        print("=" * 80)

        return

    for index, checkpoint_file in enumerate(
        checkpoint_files,
        start=1,
    ):

        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
        )

        print(
            f"{index}. "
            f"{os.path.basename(checkpoint_file)}"
        )

        print(
            f"   Epoch      : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"   Best Loss  : "
            f"{checkpoint['best_validation_loss']:.6f}"
        )

        print(
            f"   Saved At   : "
            f"{checkpoint['timestamp']}"
        )

        print()

    print("=" * 80)