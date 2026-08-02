import os
import shutil
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
    Save a complete training checkpoint.

    Every checkpoint is:
    1. Saved to a temporary file.
    2. Verified.
    3. Atomically renamed.
    4. Backed up to Google Drive.
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

    versioned_tmp = (
        versioned_checkpoint + ".tmp"
    )

    latest_tmp = (
        latest_checkpoint + ".tmp"
    )

    # --------------------------------------------------
    # Save temporary versioned checkpoint
    # --------------------------------------------------

    torch.save(
        checkpoint,
        versioned_tmp,
    )

    if not verify_checkpoint(
        versioned_tmp,
    ):

        if os.path.exists(
            versioned_tmp,
        ):
            os.remove(
                versioned_tmp,
            )

        raise RuntimeError(
            "Versioned checkpoint verification failed."
        )

    os.replace(
        versioned_tmp,
        versioned_checkpoint,
    )

    # --------------------------------------------------
    # Save temporary latest checkpoint
    # --------------------------------------------------

    torch.save(
        checkpoint,
        latest_tmp,
    )

    if not verify_checkpoint(
        latest_tmp,
    ):

        if os.path.exists(
            latest_tmp,
        ):
            os.remove(
                latest_tmp,
            )

        raise RuntimeError(
            "Latest checkpoint verification failed."
        )

    os.replace(
        latest_tmp,
        latest_checkpoint,
    )

    # --------------------------------------------------
    # Final verification
    # --------------------------------------------------

    if not verify_checkpoint(
        versioned_checkpoint,
    ):
        raise RuntimeError(
            "Final verification failed "
            "for versioned checkpoint."
        )

    if not verify_checkpoint(
        latest_checkpoint,
    ):
        raise RuntimeError(
            "Final verification failed "
            "for latest checkpoint."
        )

    # --------------------------------------------------
    # Backup to Google Drive
    # --------------------------------------------------

    backup_checkpoint_to_drive(
        versioned_checkpoint,
    )

    backup_checkpoint_to_drive(
        latest_checkpoint,
    )

    print("=" * 80)
    print("CHECKPOINT SAVED")
    print("=" * 80)
    print(
        f"Epoch    : {epoch}"
    )
    print(
        f"Latest   : {latest_checkpoint}"
    )
    print(
        f"Version  : {versioned_checkpoint}"
    )
    print("=" * 80)

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

    backup_best_model_to_drive()

    print(
        "\nNew best model saved "
        f"(Validation Loss: {validation_loss:.6f})"
    )

    print(
        f"Location: {best_model_directory}"
    )

def backup_lora_adapter_to_drive() -> None:
    """
    Backup the exported LoRA adapter to Google Drive.
    """

    drive_root = "/content/drive/MyDrive"

    if not os.path.exists(
        drive_root,
    ):

        print(
            "Google Drive is not mounted. "
            "Skipping LoRA backup."
        )

        return

    config = load_configs()

    source_directory = (
        config["checkpoint"][
            "lora_export_directory"
        ]
    )

    if not os.path.exists(
        source_directory,
    ):
        print(
            "LoRA adapter directory not found."
        )

        return

    destination_directory = os.path.join(
        drive_root,
        "Financial-LoRA",
        "artifacts",
        "lora_adapter",
    )

    if os.path.exists(
        destination_directory,
    ):

        shutil.rmtree(
            destination_directory,
        )

    shutil.copytree(
        source_directory,
        destination_directory,
    )

    print(
        "LoRA adapter backed up to Google Drive."
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


def backup_checkpoint_to_drive(
    source_path: str,
) -> None:
    """
    Copy a verified checkpoint to Google Drive and
    verify that the copied checkpoint is valid.
    """

    drive_root = "/content/drive/MyDrive"

    if not os.path.exists(
        drive_root,
    ):

        print(
            "Google Drive is not mounted. "
            "Skipping backup."
        )

        return

    backup_directory = os.path.join(
        drive_root,
        "Financial-LoRA",
        "artifacts",
        "checkpoints",
    )

    os.makedirs(
        backup_directory,
        exist_ok=True,
    )

    destination_path = os.path.join(
        backup_directory,
        os.path.basename(
            source_path,
        ),
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    if not os.path.exists(
        destination_path,
    ):

        raise RuntimeError(
            f"Google Drive backup failed:\n"
            f"{destination_path}"
        )

    source_size = os.path.getsize(
        source_path,
    )

    destination_size = os.path.getsize(
        destination_path,
    )

    if source_size != destination_size:

        raise RuntimeError(
            "Google Drive backup is incomplete."
        )

    source_checkpoint = torch.load(
        source_path,
        map_location="cpu",
    )

    destination_checkpoint = torch.load(
        destination_path,
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
        - destination_checkpoint.keys()
    )

    if missing_keys:

        raise RuntimeError(
            "Drive checkpoint is corrupted.\n"
            f"Missing keys: {missing_keys}"
        )

    if (
        source_checkpoint["epoch"]
        != destination_checkpoint["epoch"]
    ):

        raise RuntimeError(
            "Drive checkpoint epoch mismatch."
        )

    print("=" * 80)
    print("GOOGLE DRIVE BACKUP VERIFIED")
    print("=" * 80)
    print(
        f"Checkpoint : {os.path.basename(source_path)}"
    )
    print(
        f"Epoch      : "
        f"{destination_checkpoint['epoch']}"
    )
    print(
        f"Location   : "
        f"{destination_path}"
    )
    print("=" * 80)


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

        print(
            f"{index}. "
            f"{os.path.basename(checkpoint_file)}"
        )

        try:

            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
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

        except Exception:

            print(
                "   Status     : CORRUPTED"
            )

        print()

    print("=" * 80)


def backup_best_model_to_drive() -> None:
    """
    Backup the best LoRA adapter to Google Drive.
    """

    drive_root = "/content/drive/MyDrive"

    if not os.path.exists(
        drive_root,
    ):

        print(
            "Google Drive is not mounted. "
            "Skipping best model backup."
        )

        return

    config = load_configs()

    source_directory = (
        config["checkpoint"][
            "best_model_directory"
        ]
    )

    if not os.path.exists(
        source_directory,
    ):
        return

    destination_directory = os.path.join(
        drive_root,
        "Financial-LoRA",
        "artifacts",
        "best_model",
    )

    if os.path.exists(
        destination_directory,
    ):
        shutil.rmtree(
            destination_directory,
        )

    shutil.copytree(
        source_directory,
        destination_directory,
    )

    print(
        "Best model backed up to Drive."
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
        config["checkpoint"][
            "lora_export_directory"
        ]
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

    backup_lora_adapter_to_drive()

    print(
        "\nLoRA adapter exported successfully."
    )

    print(
        f"Location: {export_directory}"
    )