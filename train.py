"""
Production entry point for the QLoRA training pipeline.
"""

from src.training.tokenizer import get_tokenized_dataset
from src.training.trainer import train


def main() -> None:
    """
    Load the complete tokenized dataset and start training.
    """

    print("=" * 80)
    print("FINANCIAL LORA FINE-TUNING")
    print("=" * 80)

    tokenized_dataset = get_tokenized_dataset()

    if not isinstance(
        tokenized_dataset,
        dict,
    ):
        raise RuntimeError(
            "Tokenization pipeline did not return a dictionary."
        )

    required_splits = {
        "train",
    }

    missing_splits = sorted(
        required_splits
        - set(tokenized_dataset.keys())
    )

    if missing_splits:
        raise RuntimeError(
            "Tokenized dataset is missing required splits: "
            f"{missing_splits}"
        )

    for split_name, split_data in tokenized_dataset.items():

        if not isinstance(
            split_data,
            list,
        ):
            raise RuntimeError(
                f"Tokenized split '{split_name}' must be a list."
            )

        if not split_data:
            raise RuntimeError(
                f"Tokenized split '{split_name}' is empty."
            )

        print(
            f"{split_name:12}: {len(split_data):,} samples"
        )

    print("=" * 80)
    print("Starting training...")
    print("=" * 80)

    final_state = train(
        tokenized_dataset=tokenized_dataset
    )

    if not isinstance(
        final_state,
        dict,
    ):
        raise RuntimeError(
            "Training did not return a valid training state."
        )

    print("=" * 80)
    print("TRAINING ENTRY POINT FINISHED")
    print("=" * 80)

    print(
        f"Completed Epochs : {final_state.get('epoch')}"
    )

    print(
        f"Global Steps     : {final_state.get('global_step')}"
    )

    print(
        f"Final Train Loss : {final_state.get('train_loss')}"
    )

    print(
        f"Final Val Loss   : {final_state.get('validation_loss')}"
    )

    print(
        f"Best Val Loss    : {final_state.get('best_metric')}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()