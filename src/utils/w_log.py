import logging
import os

import wandb

from src.utils.config_loader import load_configs

def initialize_logger() -> None:
    """
    Initialize the local training logger.
    """

    config = load_configs()

    logging_config = config["logging"]

    log_directory = (
        logging_config["log_directory"]
    )

    os.makedirs(
        log_directory,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_directory,
        logging_config["file_name"],
    )

    logging.basicConfig(
        filename=log_file,
        level=getattr(
            logging,
            logging_config["level"],
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        force=True,
    )

    print(
        f"Logging to: {log_file}"
    )


def initialize_wandb() -> None:
    """
    Initialize logging and W&B.
    """

    initialize_logger()

    config = load_configs()

    wandb_config = config["wandb"]

    if not wandb_config["enabled"]:

        logging.info(
            "Weights & Biases disabled."
        )

        print(
            "W&B is disabled."
        )

        return

    if wandb.run is not None:

        logging.info(
            "W&B already initialized."
        )

        print(
            "W&B run already exists."
        )

        return

    wandb.init(
        project=wandb_config["project"],
        name=wandb_config["run_name"],
        tags=wandb_config["tags"],
        notes=wandb_config["notes"],
        config=config,
    )

    logging.info(
        "Weights & Biases initialized."
    )

    print(
        "Weights & Biases initialized."
    )


def log_metrics(
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
    gradient_norm: float,
    global_step: int,
) -> None:
    """
    Log metrics locally and to W&B.
    """

    logging.info(
        f"Epoch={epoch} | "
        f"GlobalStep={global_step} | "
        f"TrainLoss={train_loss:.6f} | "
        f"ValidationLoss={validation_loss:.6f} | "
        f"LearningRate={learning_rate:.8f} | "
        f"GradientNorm={gradient_norm:.6f}"
    )

    config = load_configs()

    if not config["wandb"]["enabled"]:

        return

    if wandb.run is None:

        return

    wandb.log(
        {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }
    )

def finish_wandb() -> None:
    """
    Finish the current W&B run.
    """

    logging.info(
        "Training finished."
    )

    config = load_configs()

    if not config["wandb"]["enabled"]:

        return

    if wandb.run is None:

        return

    wandb.finish()

    print(
        "Weights & Biases run finished."
    )