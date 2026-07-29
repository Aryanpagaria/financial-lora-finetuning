from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from src.training.dataset import get_dataloaders
from src.training.model import get_model
from src.utils.config_loader import load_configs


def get_device() -> torch.device:
    """Return the available computation device."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    return device


def create_optimizer(model: Any) -> AdamW:
    """Create the optimizer for training."""

    config = load_configs()

    learning_rate = config["training"]["learning_rate"]

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
    )

    return optimizer


def create_scheduler(optimizer: AdamW,train_dataloader: Any,) -> Any:
    """Create the learning rate scheduler."""

    config = load_configs()

    epochs = config["training"]["epochs"]
    warmup_steps = config["training"]["warmup_steps"]

    total_training_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    return scheduler

def move_batch_to_device(batch: dict,device: torch.device,) -> dict:
    """Move a batch to the selected device."""

    return {
        key: value.to(device)
        for key, value in batch.items()
    }

def load_training_components():
    """Load all components required for training."""

    
    device = get_device()

    
    dataloaders = get_dataloaders()

    
    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["validation"]

    
    model = get_model()

    print("
    model.to(device)

    
    optimizer = create_optimizer(model)

    
    scheduler = create_scheduler(
        optimizer,
        train_dataloader,
    )

    

    return (
        device,
        train_dataloader,
        validation_dataloader,
        model,
        optimizer,
        scheduler,
    )

def train_one_epoch(
    model: Any,
    train_dataloader: Any,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        train_dataloader,
        desc="Training",
    )

    for batch in progress_bar:

        batch = move_batch_to_device(
            batch,
            device,
        )

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / len(train_dataloader)

    return average_loss

def validate(
    model: Any,
    validation_dataloader: Any,
    device: torch.device,
) -> float:
    """Evaluate the model on the validation dataset."""

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in validation_dataloader:

            batch = move_batch_to_device(
                batch,
                device,
            )

            outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()

    average_loss = total_loss / len(validation_dataloader)

    return average_loss



def save_checkpoint(model: Any,epoch: int,) -> None:
    """Save the LoRA model checkpoint."""

    config = load_configs()

    output_dir = Path(config["evaluation"]["output_directory"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"

    model.save_pretrained(checkpoint_dir)

    print(f"Checkpoint saved to: {checkpoint_dir}")

def train() -> None:
    """Train the LoRA model."""
    print("Entered train()")
    config = load_configs()

    epochs = config["training"]["epochs"]

    (
        device,
        train_dataloader,
        validation_dataloader,
        model,
        optimizer,
        scheduler,
    ) = load_training_components()

    print(f"Training on: {device}")

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        validation_loss = validate(
            model=model,
            validation_dataloader=validation_dataloader,
            device=device,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            epoch=epoch + 1,
        )

    print("Training completed.")




