from typing import Any
from src.utils.config_loader import load_configs
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from src.training.tokenizer import get_tokenized_dataset

class FinancialDataset(Dataset):
    """PyTorch Dataset for tokenized financial QA samples."""

    def __init__(self,samples: list[dict[str, Any]],) -> None:
        self.samples = samples

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self,index: int,) -> dict[str, torch.Tensor]:
        
        sample = self.samples[index]

        return {
            "input_ids": torch.tensor(
                sample["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                sample["attention_mask"],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.long,
            ),
        }





def create_dataloader(dataset: FinancialDataset,batch_size: int,shuffle: bool,) -> DataLoader:
    """Create a PyTorch DataLoader."""

    return DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    pin_memory=True,
)



def get_dataloaders() -> dict[str, DataLoader]:
    """Return train, validation and test dataloaders."""

    config = load_configs()

    tokenized_dataset = get_tokenized_dataset()

    batch_size = config["training"]["batch_size"]

    train_dataset = FinancialDataset(
    tokenized_dataset["train"][:100]
)

    validation_dataset = FinancialDataset(
    tokenized_dataset["validation"][:20]
)

    test_dataset = FinancialDataset(
        tokenized_dataset["test"]
    )

    return {
    "train": create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    ),
    "validation": create_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    ),
    "test": create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    ),
}


if __name__ == "__main__":

    dataloaders = get_dataloaders()

    train_loader = dataloaders["train"]

    batch = next(iter(train_loader))

    print("=" * 80)
    print("DATALOADER REPORT")
    print("=" * 80)

    print(f"Batch Size : {batch['input_ids'].shape[0]}")
    print()

    print(f"Input IDs Shape : {batch['input_ids'].shape}")
    print(f"Attention Mask Shape : {batch['attention_mask'].shape}")
    print(f"Labels Shape : {batch['labels'].shape}")