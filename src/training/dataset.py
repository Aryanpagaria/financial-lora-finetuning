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
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
        persistent_workers=torch.cuda.is_available(),
    )



def get_dataloaders() -> dict[str, DataLoader]:
    """
    Return train, validation and test dataloaders.
    """

    config = load_configs()

    tokenized_dataset = get_tokenized_dataset()

    batch_size = config["training"]["batch_size"]

    train_dataset = FinancialDataset(
        tokenized_dataset["train"]
    )

    validation_dataset = FinancialDataset(
        tokenized_dataset["validation"]
    )

    test_dataset = FinancialDataset(
        tokenized_dataset["test"]
    )

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    sanity_check_datasets(
        datasets,
    )

    print_dataset_statistics(
        datasets,
    )

    dataloaders = {
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

    print_dataloader_statistics(
        dataloaders,
    )

    preview_dataset_sample(
        train_dataset,
    )

    return dataloaders

def print_dataset_statistics(
    datasets: dict[str, FinancialDataset],
) -> None:
    """
    Print dataset statistics.
    """

    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    total_samples = 0

    for split_name, dataset in datasets.items():

        sample_count = len(dataset)

        total_samples += sample_count

        print(
            f"{split_name:<12}: {sample_count:,} samples"
        )

    print("-" * 80)

    print(
        f"Total Samples : {total_samples:,}"
    )

    print("=" * 80)



def print_dataloader_statistics(
    dataloaders: dict[str, DataLoader],
) -> None:
    """
    Print dataloader statistics.
    """

    print("=" * 80)
    print("DATALOADER STATISTICS")
    print("=" * 80)

    for split_name, dataloader in dataloaders.items():

        print(
            f"{split_name:<12}: "
            f"{len(dataloader)} batches"
        )

    print("=" * 80)



def sanity_check_datasets(
    datasets: dict[str, FinancialDataset],
) -> None:
    """
    Validate datasets before training.
    """

    for split_name, dataset in datasets.items():

        if len(dataset) == 0:

            raise RuntimeError(
                f"{split_name} dataset is empty."
            )

        sample = dataset[0]

        required_keys = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        if set(sample.keys()) != required_keys:

            raise RuntimeError(
                f"{split_name} dataset has an invalid format."
            )

    print("=" * 80)
    print("DATASET SANITY CHECK PASSED")
    print("=" * 80)




def preview_dataset_sample(
    dataset: FinancialDataset,
) -> None:
    """
    Preview one processed sample.
    """

    sample = dataset[0]

    print("=" * 80)
    print("DATASET SAMPLE")
    print("=" * 80)

    print(
        f"Input Length : "
        f"{len(sample['input_ids'])}"
    )

    print(
        f"Attention Length : "
        f"{len(sample['attention_mask'])}"
    )

    print(
        f"Label Length : "
        f"{len(sample['labels'])}"
    )

    print("=" * 80)




if __name__ == "__main__":

    dataloaders = get_dataloaders()

    train_loader = dataloaders["train"]

    batch = next(iter(train_loader))

    print("=" * 80)
    print("DATALOADER REPORT")
    print("=" * 80)

    print(f"Batch Size : {batch['input_ids'].shape[0]}")
    print()

    print(f"Input IDs Shape      : {batch['input_ids'].shape}")
    print(f"Attention Mask Shape : {batch['attention_mask'].shape}")
    print(f"Labels Shape         : {batch['labels'].shape}")