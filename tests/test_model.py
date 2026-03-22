"""
tests/test_model.py — Integration tests for model.py pipelines.

These tests download and run real HuggingFace models.
They are marked 'slow' and skipped by default.

Run all tests including slow:
    pytest tests/test_model.py -v -m slow

Run only fast tests (skips this file):
    pytest tests/ -v -m "not slow"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sentiment_model():
    from model import predict_sentiment
    return predict_sentiment


@pytest.fixture(scope="module")
def category_model():
    from model import predict_category
    return predict_category


@pytest.fixture(scope="module")
def aspect_model():
    from model import predict_aspect
    return predict_aspect


# ---------------------------------------------------------------------------
# predict_sentiment
# ---------------------------------------------------------------------------

class TestPredictSentiment:

    def test_clearly_positive_review(self, sentiment_model):
        result = sentiment_model("This is an absolutely amazing product, I love it!")
        assert result["label"] == "POSITIVE"
        assert result["score"] > 0.9

    def test_clearly_negative_review(self, sentiment_model):
        result = sentiment_model("Terrible product, broke after one day. Complete waste of money.")
        assert result["label"] == "NEGATIVE"
        assert result["score"] > 0.9

    def test_returns_required_keys(self, sentiment_model):
        result = sentiment_model("Decent product.")
        assert "label" in result
        assert "score" in result
        assert "emoji" in result

    def test_score_between_0_and_1(self, sentiment_model):
        result = sentiment_model("It was okay.")
        assert 0.0 <= result["score"] <= 1.0

    def test_label_is_valid(self, sentiment_model):
        result = sentiment_model("Good enough.")
        assert result["label"] in ("POSITIVE", "NEGATIVE")

    def test_emoji_positive(self, sentiment_model):
        result = sentiment_model("Best purchase ever, highly recommend!")
        assert result["emoji"] == "😊"

    def test_emoji_negative(self, sentiment_model):
        result = sentiment_model("Worst product I have ever bought.")
        assert result["emoji"] == "😞"

    def test_empty_string_returns_default(self, sentiment_model):
        result = sentiment_model("")
        assert "label" in result
        assert "score" in result

    def test_long_text_truncated_safely(self, sentiment_model):
        long_text = "Great product. " * 200
        result = sentiment_model(long_text)
        assert result["label"] in ("POSITIVE", "NEGATIVE")


# ---------------------------------------------------------------------------
# predict_category
# ---------------------------------------------------------------------------

class TestPredictCategory:

    def test_smartphone_reviews(self, category_model):
        text = (
            "This phone has incredible battery life and a stunning camera. "
            "The Android performance is smooth and the display is vivid."
        )
        result = category_model(text)
        assert result == "smartphone"

    def test_laptop_reviews(self, category_model):
        text = (
            "This laptop is fast for coding. The keyboard is comfortable "
            "and battery lasts 8 hours. Great for work and programming."
        )
        result = category_model(text)
        assert result == "laptop"

    def test_headphones_reviews(self, category_model):
        text = (
            "These headphones have incredible sound quality and the noise "
            "cancellation is top notch. Bass is deep and treble is crisp."
        )
        result = category_model(text)
        assert result == "headphones"

    def test_returns_string(self, category_model):
        result = category_model("Good product overall.")
        assert isinstance(result, str)

    def test_returns_valid_category(self, category_model):
        from model import CATEGORY_LABELS
        result = category_model("Great device with good battery.")
        assert result in CATEGORY_LABELS

    def test_empty_returns_other(self, category_model):
        result = category_model("")
        assert result == "other"


# ---------------------------------------------------------------------------
# predict_aspect
# ---------------------------------------------------------------------------

class TestPredictAspect:

    def test_returns_dict(self, aspect_model):
        result = aspect_model("Great battery life and stunning camera quality.")
        assert isinstance(result, dict)

    def test_returns_all_default_aspects(self, aspect_model):
        from model import DEFAULT_ASPECTS
        result = aspect_model("Good product.")
        for aspect in DEFAULT_ASPECTS:
            assert aspect in result

    def test_scores_between_0_and_1(self, aspect_model):
        result = aspect_model("Amazing battery life.")
        for aspect, score in result.items():
            assert 0.0 <= score <= 1.0, f"Score out of range for {aspect}: {score}"

    def test_battery_review_scores_battery_highly(self, aspect_model):
        result = aspect_model(
            "The battery life is absolutely incredible, lasts two full days."
        )
        assert result["battery life"] > result["customer service"]

    def test_custom_aspects(self, aspect_model):
        custom = ["color", "weight", "packaging"]
        result = aspect_model("Great product.", aspects=custom)
        assert set(result.keys()) == set(custom)

    def test_empty_text_returns_zeros(self, aspect_model):
        result = aspect_model("")
        for score in result.values():
            assert score == 0.0
