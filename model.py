"""
model.py — AI pipeline definitions for Product Insight AI.

Loads three HuggingFace pipelines (lazily, with caching) and exposes
three prediction helpers used throughout the project.

Fine-tuned model
----------------
If ./fine_tuned_model/ exists (created by train.py), it is used
automatically for sentiment analysis instead of the default SST-2
checkpoint.  Set USE_FINETUNED = False to force the baseline.
"""

import os
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Config — switch between baseline and fine-tuned model
# ---------------------------------------------------------------------------

_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_FINETUNED_DIR  = os.path.join(_BASE_DIR, "fine_tuned_model")
_BASELINE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Auto-detect: use fine-tuned if the directory exists and has model files
USE_FINETUNED = (
    os.path.isdir(_FINETUNED_DIR)
    and os.path.exists(os.path.join(_FINETUNED_DIR, "config.json"))
)

SENTIMENT_MODEL = _FINETUNED_DIR if USE_FINETUNED else _BASELINE_MODEL


# ---------------------------------------------------------------------------
# Pipeline loaders (cached so they are instantiated only once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    """Return a cached sentiment-analysis pipeline.

    Uses the fine-tuned model from ./fine_tuned_model/ when available,
    otherwise falls back to the pre-trained SST-2 checkpoint.
    """
    from transformers import pipeline
    source = "fine-tuned" if USE_FINETUNED else "baseline (SST-2)"
    print(f"[model] Loading sentiment pipeline ({source})…")
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
    )


@lru_cache(maxsize=1)
def _get_zero_shot_pipeline():
    """Return a cached BART zero-shot-classification pipeline."""
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

CATEGORY_LABELS = [
    "smartphone",
    "laptop",
    "headphones",
    "tablet",
    "smartwatch",
    "camera",
    "TV",
    "other",
]

DEFAULT_ASPECTS = [
    "battery life",
    "camera quality",
    "display",
    "performance",
    "price",
    "design",
    "sound quality",
    "build quality",
    "software",
    "customer service",
]

_SENTIMENT_EMOJI = {"POSITIVE": "😊", "NEGATIVE": "😞"}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_sentiment(text: str) -> dict:
    """
    Run sentiment analysis on *text*.

    Returns
    -------
    dict with keys:
        label  : "POSITIVE" or "NEGATIVE"
        score  : float in [0, 1]
        emoji  : corresponding emoji string
    """
    if not text or not text.strip():
        return {"label": "POSITIVE", "score": 0.5, "emoji": "😐"}

    pipe = _get_sentiment_pipeline()
    # Truncate to 512 tokens to stay within model limits
    result = pipe(text[:1024], truncation=True, max_length=512)[0]
    label = result["label"]  # "POSITIVE" | "NEGATIVE"
    score = result["score"]
    return {
        "label": label,
        "score": score,
        "emoji": _SENTIMENT_EMOJI.get(label, "😐"),
    }


def predict_category(text: str, labels: Optional[list] = None) -> str:
    """
    Detect the most likely product category for *text* via zero-shot
    classification.

    Parameters
    ----------
    text   : review text (or concatenated reviews)
    labels : candidate category labels; defaults to CATEGORY_LABELS

    Returns
    -------
    Top-scoring category string (e.g. "smartphone").
    """
    if not text or not text.strip():
        return "other"

    candidate_labels = labels or CATEGORY_LABELS
    pipe = _get_zero_shot_pipeline()
    result = pipe(
        text[:1024],
        candidate_labels=candidate_labels,
        hypothesis_template="This review is about a {}.",
        truncation=True,
    )
    return result["labels"][0]


def predict_aspect(text: str, aspects: Optional[list] = None) -> dict:
    """
    Score each *aspect* against *text* using zero-shot classification.

    The model is asked to classify the text under each aspect as a label;
    we interpret the probability returned by MNLI for each aspect as a
    relevance/sentiment score.

    Parameters
    ----------
    text    : single review text
    aspects : list of aspect strings; defaults to DEFAULT_ASPECTS

    Returns
    -------
    dict mapping aspect string → score (float in [0, 1])
    """
    if not text or not text.strip():
        return {a: 0.0 for a in (aspects or DEFAULT_ASPECTS)}

    candidate_labels = aspects or DEFAULT_ASPECTS
    pipe = _get_zero_shot_pipeline()
    result = pipe(text[:1024], candidate_labels=candidate_labels, truncation=True)
    # result["labels"] is sorted by descending score; zip back together
    scores = dict(zip(result["labels"], result["scores"]))
    # Return in the original aspect order
    return {a: scores.get(a, 0.0) for a in candidate_labels}
