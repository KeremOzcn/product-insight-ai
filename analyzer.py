"""
analyzer.py — Core multi-review analysis engine for Product Insight AI.

Public API
----------
analyze_reviews(reviews: list[str]) -> dict

The returned dict contains overall sentiment, detected product category,
aspect scores, topic frequencies, a generated summary, and per-review
breakdowns.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from model import (
    DEFAULT_ASPECTS,
    predict_aspect,
    predict_category,
    predict_sentiment,
)

# ---------------------------------------------------------------------------
# Topic keyword mapping used for frequency analysis
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "battery":     ["battery", "charge", "charging", "power", "mah", "drain"],
    "camera":      ["camera", "photo", "picture", "image", "lens", "megapixel", "shot", "selfie"],
    "display":     ["screen", "display", "resolution", "brightness", "oled", "amoled", "lcd", "panel"],
    "performance": ["fast", "slow", "lag", "processor", "speed", "performance", "cpu", "gpu", "smooth"],
    "price":       ["price", "cost", "expensive", "cheap", "worth", "value", "money", "affordable"],
    "design":      ["design", "look", "feel", "build", "quality", "material", "slim", "premium", "aesthetic"],
    "sound":       ["sound", "audio", "speaker", "volume", "bass", "treble", "noise", "microphone"],
    "software":    ["software", "app", "update", "os", "interface", "ui", "ux", "bug", "crash", "feature"],
}

# Number of top aspects to surface per sentiment group
_TOP_N = 3

# Keyword-based product type hints (checked before zero-shot to reduce misclassification)
# Key insight: "camera quality" in a review ≠ the product IS a camera.
# Only match product-specific terms, not feature words.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "smartphone":  ["phone", "iphone", "android", "mobile", "samsung", "pixel",
                    "oneplus", "xiaomi", "huawei", "smartphone"],
    "laptop":      ["laptop", "notebook", "macbook", "ultrabook", "chromebook",
                    "thinkpad", "surface pro"],
    "headphones":  ["headphone", "earphone", "earbud", "headset", "airpod",
                    "earbuds", "in-ear", "over-ear"],
    "tablet":      ["tablet", "ipad", "kindle", "e-reader"],
    "smartwatch":  ["smartwatch", "watch", "fitbit", "garmin", "apple watch"],
    "TV":          ["television", " tv ", "smart tv", "oled tv", "qled", "4k tv"],
    "camera":      ["dslr", "mirrorless", "point-and-shoot", "gopro", "action cam",
                    "camera body", "lens kit"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_topics(reviews: list[str]) -> Counter:
    """Return a Counter of topic hits across all review texts."""
    topic_counter: Counter = Counter()
    for review in reviews:
        lower = review.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                topic_counter[topic] += 1
    return topic_counter


def _aggregate_aspects(per_review: list[dict]) -> dict[str, float]:
    """Average aspect scores across all reviews."""
    if not per_review:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in per_review:
        for aspect, score in result.get("aspects", {}).items():
            totals[aspect] = totals.get(aspect, 0.0) + score
            counts[aspect] = counts.get(aspect, 0) + 1
    return {a: totals[a] / counts[a] for a in totals}


def _classify_aspects(per_review: list[dict]) -> tuple[list[str], list[str]]:
    """
    Derive top positive and negative aspects using sentiment-weighted scoring.

    For each aspect we average its zero-shot score separately across:
      - POSITIVE reviews  → high avg score means users praise this aspect
      - NEGATIVE reviews  → high avg score means users complain about this aspect

    This prevents the naive approach of just taking top/bottom globally,
    which would wrongly label praised aspects as negative.
    """
    def _avg(reviews: list[dict]) -> dict[str, float]:
        if not reviews:
            return {}
        totals: dict[str, float] = {}
        for r in reviews:
            for aspect, score in r.get("aspects", {}).items():
                totals[aspect] = totals.get(aspect, 0.0) + score
        return {a: v / len(reviews) for a, v in totals.items()}

    pos_reviews = [r for r in per_review if r["sentiment"] == "Positive"]
    neg_reviews = [r for r in per_review if r["sentiment"] == "Negative"]

    pos_scores = _avg(pos_reviews) or _avg(per_review)
    neg_scores = _avg(neg_reviews)

    # Top positive: highest-scoring aspects in positive reviews
    top_positive = [
        a for a, _ in sorted(pos_scores.items(), key=lambda x: x[1], reverse=True)
    ][:_TOP_N]

    if neg_scores:
        # Top negative: highest-scoring aspects in negative reviews, skip overlap
        top_negative = [
            a for a, _ in sorted(neg_scores.items(), key=lambda x: x[1], reverse=True)
            if a not in top_positive
        ][:_TOP_N]
    else:
        # No negative reviews: use LOWEST-scoring aspects from positive reviews
        # as proxy for under-praised / weak features
        top_negative = [
            a for a, _ in sorted(pos_scores.items(), key=lambda x: x[1])
            if a not in top_positive
        ][:_TOP_N]

    return top_positive, top_negative


def _keyword_category(text: str) -> Optional[str]:
    """
    Fast keyword pre-check before calling the zero-shot model.

    Returns the matched category name or None if no strong match found.
    This prevents feature words like 'camera quality' from being mistaken
    for the product being a camera.
    """
    lower = text.lower()
    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[category] = hits
    if not scores:
        return None
    best = max(scores, key=lambda c: scores[c])
    # Only trust keyword result if it has a clear lead
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) == 1 or sorted_scores[0] > sorted_scores[1]:
        return best
    return None


def _determine_overall_sentiment(
    pos_count: int, neg_count: int
) -> tuple[str, float]:
    """
    Derive the overall sentiment label and aggregate confidence.

    Returns
    -------
    (label, score) where label is "Positive" | "Negative" | "Mixed"
    and score is in [0, 1].
    """
    total = pos_count + neg_count
    if total == 0:
        return "Mixed", 0.5

    pos_ratio = pos_count / total
    if pos_ratio >= 0.65:
        return "Positive", pos_ratio
    elif pos_ratio <= 0.35:
        return "Negative", 1.0 - pos_ratio
    else:
        # Mixed — confidence is how far from 50/50
        confidence = abs(pos_ratio - 0.5) * 2  # 0 at perfect tie, 1 at 65/35
        return "Mixed", max(0.5, confidence + 0.45)


def _build_summary(
    n: int,
    category: str,
    overall: str,
    score: float,
    pos_aspects: list[str],
    neg_aspects: list[str],
) -> str:
    """Build a concise 3–4 sentence summary string (no LLM required)."""
    pos_str = ", ".join(pos_aspects) if pos_aspects else "overall quality"
    neg_str = ", ".join(neg_aspects) if neg_aspects else "minor issues"
    return (
        f"Analyzed {n} review{'s' if n != 1 else ''} for a {category}. "
        f"Overall sentiment is {overall} ({score:.0%} confidence). "
        f"Users frequently praise: {pos_str}. "
        f"Common complaints include: {neg_str}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_reviews(reviews: list[str]) -> dict:
    """
    Run full analysis on a list of review strings.

    Parameters
    ----------
    reviews : list of 1–100 review strings

    Returns
    -------
    dict with keys:
        overall_sentiment      : "Positive" | "Negative" | "Mixed"
        sentiment_score        : float in [0, 1]
        positive_count         : int
        negative_count         : int
        product_category       : str
        top_positive_aspects   : list[str]
        top_negative_aspects   : list[str]
        most_discussed_topics  : list[tuple(str, int)]  — top 5
        summary                : str
        per_review_results     : list[dict]
    """
    if not reviews:
        raise ValueError("At least one review is required.")

    # Clamp to 100 reviews
    reviews = [r.strip() for r in reviews if r.strip()][:100]

    # ------------------------------------------------------------------
    # 1. Per-review sentiment + aspect scoring
    # ------------------------------------------------------------------
    per_review_results: list[dict] = []
    pos_count = 0
    neg_count = 0

    for idx, review in enumerate(reviews):
        sentiment = predict_sentiment(review)
        aspects = predict_aspect(review)

        if sentiment["label"] == "POSITIVE":
            pos_count += 1
        else:
            neg_count += 1

        per_review_results.append(
            {
                "index": idx + 1,
                "review_snippet": review[:120] + ("…" if len(review) > 120 else ""),
                "sentiment": sentiment["label"].title(),
                "score": round(sentiment["score"], 4),
                "emoji": sentiment["emoji"],
                "aspects": aspects,
            }
        )

    # ------------------------------------------------------------------
    # 2. Product category detection
    #    Fast keyword check first; fall back to zero-shot only if needed.
    # ------------------------------------------------------------------
    sample_text = " ".join(reviews[:10])
    product_category = _keyword_category(sample_text) or predict_category(sample_text)

    # ------------------------------------------------------------------
    # 3. Aggregate aspect scores + sentiment-weighted classification
    # ------------------------------------------------------------------
    agg_aspects = _aggregate_aspects(per_review_results)
    top_positive_aspects, top_negative_aspects = _classify_aspects(per_review_results)

    # ------------------------------------------------------------------
    # 4. Topic frequency
    # ------------------------------------------------------------------
    topic_counter = _count_topics(reviews)
    most_discussed_topics = topic_counter.most_common(5)

    # ------------------------------------------------------------------
    # 5. Overall sentiment
    # ------------------------------------------------------------------
    overall_sentiment, sentiment_score = _determine_overall_sentiment(
        pos_count, neg_count
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    summary = _build_summary(
        len(reviews),
        product_category,
        overall_sentiment,
        sentiment_score,
        top_positive_aspects,
        top_negative_aspects,
    )

    return {
        "overall_sentiment": overall_sentiment,
        "sentiment_score": round(sentiment_score, 4),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "product_category": product_category,
        "top_positive_aspects": top_positive_aspects,
        "top_negative_aspects": top_negative_aspects,
        "most_discussed_topics": most_discussed_topics,
        "summary": summary,
        "per_review_results": per_review_results,
        "aggregated_aspects": agg_aspects,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_REVIEWS = [
        "The battery life on this phone is incredible — lasted two full days!",
        "Camera is okay but the software keeps crashing after the last update.",
        "Great value for the price. Performance is smooth and the display is vivid.",
        "Build quality feels cheap. The charging speed is disappointing.",
        "Best smartphone I've owned. Sound quality from the speakers is top-notch.",
    ]

    print("Running analyzer smoke-test with 5 sample reviews…\n")
    result = analyze_reviews(SAMPLE_REVIEWS)

    print(f"Overall Sentiment : {result['overall_sentiment']} ({result['sentiment_score']:.0%})")
    print(f"Product Category  : {result['product_category']}")
    print(f"Positive / Negative: {result['positive_count']} / {result['negative_count']}")
    print(f"Top Positive Aspects: {result['top_positive_aspects']}")
    print(f"Top Negative Aspects: {result['top_negative_aspects']}")
    print(f"Most Discussed Topics: {result['most_discussed_topics']}")
    print(f"\nSummary:\n{result['summary']}")
