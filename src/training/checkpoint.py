import os
import torch

from src.utils.config_loader import load_configs


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    loss: float,
) -> None:
    """
    Save a training checkpoint.
    """

    config = load_configs()

    checkpoint_dir = config["checkpoint"]["save_directory"]
    checkpoint_name = config["checkpoint"]["file_name"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint_name,
    )

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(f"\nCheckpoint saved to {checkpoint_path}")




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

    print(f"Loaded checkpoint: {checkpoint_path}")

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "start_epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "best_loss": checkpoint["loss"],
    }