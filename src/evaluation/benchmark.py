"""
Benchmark multiple models using the evaluator.
"""

from typing import Any

from src.evaluation.evaluator import evaluate


def benchmark_models(
    models: dict[str, Any],
    dataloader: Any,
    device: Any,
) -> dict:
    """
    Evaluate multiple models and compare their performance.
    """

    benchmark_results = {}

    for model_name, model in models.items():

        print(f"\nEvaluating: {model_name}")

        metrics = evaluate(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        benchmark_results[model_name] = metrics

    return benchmark_results


def print_benchmark_results(
    results: dict,
) -> None:
    """
    Display benchmark results.
    """

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    sorted_results = sorted(
        results.items(),
        key=lambda item: item[1]["loss"],
    )

    for rank, (model_name, metrics) in enumerate(
        sorted_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{model_name:<20}"
            f"Loss : {metrics['loss']:.4f}"
        )

    print("=" * 80)