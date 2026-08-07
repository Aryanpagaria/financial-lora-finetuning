def main() -> None:

    config = load_inference_config()

    tokenizer = load_tokenizer(
        config,
    )

    model = load_model(
        config,
    )

    model = load_lora_adapter(
        model,
        config,
    )

    interactive_chat(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )