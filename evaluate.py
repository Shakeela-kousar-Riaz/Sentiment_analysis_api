
import argparse
import time

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

from app.sentiment import analyze_text


LABEL_NAMES = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the sentiment model on TweetEval."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of test examples to evaluate; use 0 for all.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")

    dataset = load_dataset(
        "tweet_eval",
        "sentiment",
        split="test",
    )

    if args.limit:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    y_true = []
    y_pred = []
    latencies = []

    for index, row in enumerate(dataset, start=1):
        text = row["text"]
        true_label = LABEL_NAMES[int(row["label"])]

        start = time.perf_counter()
        result = analyze_text(text)
        elapsed = time.perf_counter() - start

        y_true.append(true_label)
        y_pred.append(result["sentiment"])
        latencies.append(elapsed)

        if index % 100 == 0:
            print(f"Evaluated {index}/{len(dataset)} examples")

    print("\n=== Sentiment Model Evaluation ===")
    print(f"Examples evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABEL_NAMES,
            target_names=LABEL_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    average_latency = sum(latencies) / len(latencies)

    print(f"Average latency: {average_latency:.4f} seconds/text")
    print(
        "Note: latency includes per-text inference and may include "
        "model initialization on the first example."
    )


if __name__ == "__main__":
    main()