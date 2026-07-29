"""
Inference entry point.
"""

from src.training.model import get_model
from src.training.tokenizer import get_tokenizer
from src.utils.device import get_device
from src.evaluation.generation import generate_response


def main() -> None:
    """
    Run interactive inference.
    """

    device = get_device()

    model = get_model()
    model.to(device)

    tokenizer = get_tokenizer()

    print("=" * 80)
    print("Financial LoRA Chat")
    print("Type 'exit' to quit.")
    print("=" * 80)

    while True:

        prompt = input("\nYou : ")

        if prompt.lower() == "exit":
            break

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )

        print(f"\nAssistant : {response}")


if __name__ == "__main__":
    main()