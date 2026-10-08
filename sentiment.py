
from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABELS = ("negative", "neutral", "positive")


@lru_cache(maxsize=1)
def get_classifier():
    """Load the model once and reuse it across requests."""
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,
        device=-1,  # CPU; use device=0 if CUDA is configured.
    )


def normalize_label(label: str) -> str:
    """Normalize model output labels into the required API labels."""
    value = label.strip().lower()

    if value in LABELS:
        return value

    # Some model configurations expose generic LABEL_0 labels.
    if value.startswith("label_"):
        try:
            index = int(value.split("_")[-1])
            if 0 <= index < len(LABELS):
                return LABELS[index]
        except ValueError:
            pass

    raise ValueError(f"Unsupported model label: {label}")


def analyze_text(text: str) -> dict:
    classifier = get_classifier()

    words = text.split()

    # Keep chunks small enough for the model's token limit.
    chunks = [
        " ".join(words[start:start + 100])
        for start in range(0, len(words), 100)
    ]

    weighted_scores = {label: 0.0 for label in LABELS}
    total_words = 0

    for chunk in chunks:
        raw_results = classifier(
            chunk,
            truncation=True,
            max_length=512,
        )

        # With top_k=None, a single input normally returns
        # a list of score dictionaries.
        if raw_results and isinstance(raw_results[0], list):
            raw_results = raw_results[0]

        chunk_scores = {label: 0.0 for label in LABELS}

        for item in raw_results:
            label = normalize_label(item["label"])
            chunk_scores[label] = float(item["score"])

        chunk_word_count = len(chunk.split())

        for label in LABELS:
            weighted_scores[label] += (
                chunk_scores[label] * chunk_word_count
            )

        total_words += chunk_word_count

    if total_words == 0:
        raise ValueError("Text cannot be empty.")

    averaged_scores = {
        label: score / total_words
        for label, score in weighted_scores.items()
    }

    sentiment = max(averaged_scores, key=averaged_scores.get)

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(averaged_scores[sentiment], 4),
    }


def is_model_loaded() -> bool:
    return get_classifier.cache_info().currsize > 0