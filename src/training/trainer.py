from typing import Any
from src.training.tokenizer import get_tokenizer
import torch
from tqdm import tqdm
from src.evaluation.evaluator import evaluate
from src.training.dataset import get_dataloaders
from src.training.model import get_model
from src.training.optimizer import (
    get_optimizer,
    print_optimizer_info,
)
from src.training.scheduler import (
    get_lr_scheduler,
    print_scheduler_info,
)
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_best_checkpoint,
    export_lora_adapter,
    cleanup_old_checkpoints,
)

from src.utils.config_loader import load_configs
from src.utils.device import (
    get_device,
    print_device_info,
)
from src.utils.seed import (
    set_seed,
    print_seed_info,
)

from src.utils.w_log import (
    initialize_wandb,
    log_metrics,
    finish_wandb,
)

from src.training.early_stopping import EarlyStopping



def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> dict:
    """
    Move a batch to the selected device.
    """

    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def load_training_components():
    """
    Load every component required for training.
    """

    set_seed()
    print_seed_info()

    device = get_device()
    print_device_info()

    dataloaders = get_dataloaders()

    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["validation"]
    tokenizer = get_tokenizer()

    model = get_model()

    model.to(device)

    optimizer = get_optimizer(model)

    print_optimizer_info()

    config = load_configs()

    epochs = config["training"]["epochs"]

    total_training_steps = (
        len(train_dataloader)
        * epochs
    )

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_training_steps=total_training_steps,
    )

    print_scheduler_info(
        total_training_steps
    )

    return (
        device,
        train_dataloader,
        validation_dataloader,
        tokenizer,
        model,
    optimizer,
    scheduler,
)




def train_one_epoch(
    model: Any,
    train_dataloader: Any,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """

    model.train()

    config = load_configs()

    gradient_accumulation_steps = (
        config["training"]["gradient_accumulation_steps"]
    )

    total_loss = 0.0
    total_gradient_norm = 0.0

    progress_bar = tqdm(
        train_dataloader,
        desc="Training",
        leave=False,
    )

    optimizer.zero_grad()

    for step, batch in enumerate(
        progress_bar,
        start=1,
    ):

        batch = move_batch_to_device(
            batch=batch,
            device=device,
        )

        outputs = model(**batch)

        loss = (
            outputs.loss
            / gradient_accumulation_steps
        )

        loss.backward()

        if (
            step % gradient_accumulation_steps == 0
            or step == len(train_dataloader)
        ):

            trainable_parameters = list(
                filter(
                    lambda parameter: parameter.requires_grad,
                    model.parameters(),
                )
            )

            trainable_with_grad = sum(
                1
                for parameter in trainable_parameters
                if parameter.grad is not None
            )

            

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )           

            total_gradient_norm += gradient_norm.item()

            total_gradient_norm += gradient_norm.item()

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

        total_loss += (
            loss.item()
            * gradient_accumulation_steps
        )

        progress_bar.set_postfix(
            {
                "loss": (
                    f"{loss.item() * gradient_accumulation_steps:.4f}"
                ),
                "lr": (
                    f"{scheduler.get_last_lr()[0]:.2e}"
                ),
            }
        )

    average_loss = (
        total_loss
        / len(train_dataloader)
    )

    average_gradient_norm = (
        total_gradient_norm
        / max(
            1,
            len(train_dataloader)
            // gradient_accumulation_steps,
        )
    )

    return (
        average_loss,
        average_gradient_norm,
    )

def train() -> dict:
    """
    Train the LoRA model.
    """

    config = load_configs()

    epochs = config["training"]["epochs"]

    (
        device,
        train_dataloader,
        validation_dataloader,
        tokenizer,
        model,
        optimizer,
        scheduler,
    ) = load_training_components()

    resume_state = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    model = resume_state["model"]
    optimizer = resume_state["optimizer"]
    scheduler = resume_state["scheduler"]

    start_epoch = resume_state["start_epoch"]
    global_step = resume_state["global_step"]
    best_loss = resume_state["best_loss"]

    early_stopping = EarlyStopping()
    early_stopping.best_loss = best_loss

    initialize_wandb()

    print(f"\nTraining started on {device}\n")

    for epoch in range(start_epoch, epochs):

        print("=" * 80)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("=" * 80)

        (
            train_loss,
            gradient_norm,
        ) = train_one_epoch(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        )

        evaluation_metrics = evaluate(
            model=model,
            dataloader=validation_dataloader,
            device=device,
        )

        validation_loss = evaluation_metrics["loss"]
        if validation_loss < best_loss:
            best_loss = validation_loss

            save_best_checkpoint(
                model=model,
                validation_loss=validation_loss,
            )

        global_step += len(train_dataloader)

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Gradient Norm   : {gradient_norm:.4f}"
        )

        print(
            f"Validation Loss : {validation_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            best_validation_loss=best_loss,
        )
        cleanup_old_checkpoints()

        current_learning_rate = scheduler.get_last_lr()[0]

        log_metrics(
            epoch=epoch + 1,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=current_learning_rate,
            gradient_norm=gradient_norm,
        )

        should_stop = early_stopping.update(
            validation_loss
        )

        if should_stop:

            print("\nEarly stopping triggered.")

            break
    export_lora_adapter(
        model=model,
        tokenizer=tokenizer,
    )
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Validation Loss : {best_loss:.6f}")
    print(f"Completed Epochs     : {epoch + 1}")
    print(f"Global Steps         : {global_step}")
    print("=" * 80)
    finish_wandb()

    print("\nTraining Finished Successfully.")
    
    return {
        "best_validation_loss": best_loss,
        "global_step": global_step,
        "epochs_completed": epoch + 1,
    }
