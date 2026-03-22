"""
tests/test_analyzer.py — Unit tests for analyzer.py logic.

Models are mocked so these tests run fast (no GPU/download needed).

Run:
    pytest tests/test_analyzer.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Import internal helpers directly (no model calls at import time)
# ---------------------------------------------------------------------------

from analyzer import (
    CATEGORY_KEYWORDS,
    TOPIC_KEYWORDS,
    _aggregate_aspects,
    _build_summary,
    _classify_aspects,
    _count_topics,
    _determine_overall_sentiment,
    _keyword_category,
)


# ---------------------------------------------------------------------------
# _count_topics
# ---------------------------------------------------------------------------

class TestCountTopics:
    def test_battery_keyword_detected(self):
        reviews = ["The battery life is amazing and charging is fast."]
        counter = _count_topics(reviews)
        assert counter["battery"] == 1

    def test_camera_keyword_detected(self):
        reviews = ["The camera takes stunning photos and the image quality is great."]
        counter = _count_topics(reviews)
        assert counter["camera"] == 1

    def test_multiple_reviews_accumulate(self):
        reviews = ["Battery is great.", "Battery drains fast."]
        counter = _count_topics(reviews)
        assert counter["battery"] == 2

    def test_empty_reviews(self):
        counter = _count_topics([])
        assert len(counter) == 0

    def test_no_match_returns_zero(self):
        reviews = ["This is a completely unrelated sentence."]
        counter = _count_topics(reviews)
        # "quality" matches "design" keyword
        assert counter.get("battery", 0) == 0

    def test_multiple_topics_same_review(self):
        reviews = ["The battery is great and the camera is stunning."]
        counter = _count_topics(reviews)
        assert counter["battery"] >= 1
        assert counter["camera"] >= 1

    def test_case_insensitive(self):
        reviews = ["BATTERY is amazing, CAMERA is great."]
        counter = _count_topics(reviews)
        assert counter["battery"] == 1
        assert counter["camera"] == 1

    def test_returns_counter(self):
        result = _count_topics(["battery is good"])
        assert isinstance(result, Counter)


# ---------------------------------------------------------------------------
# _aggregate_aspects
# ---------------------------------------------------------------------------

class TestAggregateAspects:
    def _make_review(self, aspects: dict, sentiment="Positive"):
        return {"sentiment": sentiment, "aspects": aspects}

    def test_single_review(self):
        reviews = [self._make_review({"battery life": 0.8, "price": 0.2})]
        agg = _aggregate_aspects(reviews)
        assert abs(agg["battery life"] - 0.8) < 1e-6
        assert abs(agg["price"] - 0.2) < 1e-6

    def test_averages_two_reviews(self):
        reviews = [
            self._make_review({"battery life": 0.6}),
            self._make_review({"battery life": 0.4}),
        ]
        agg = _aggregate_aspects(reviews)
        assert abs(agg["battery life"] - 0.5) < 1e-6

    def test_empty_returns_empty(self):
        assert _aggregate_aspects([]) == {}

    def test_missing_aspect_key_ignored(self):
        reviews = [
            {"sentiment": "Positive", "aspects": {"battery life": 0.7}},
            {"sentiment": "Negative", "aspects": {"price": 0.5}},
        ]
        agg = _aggregate_aspects(reviews)
        assert "battery life" in agg
        assert "price" in agg


# ---------------------------------------------------------------------------
# _classify_aspects
# ---------------------------------------------------------------------------

class TestClassifyAspects:
    def _make_review(self, sentiment, aspects):
        return {"sentiment": sentiment, "aspects": aspects, "score": 0.9}

    def test_positive_aspects_from_positive_reviews(self):
        # 6 aspects: top 3 praised in positive reviews, top 3 complained in negative reviews
        pos_aspects = {"battery life": 0.9, "performance": 0.8, "display": 0.7,
                       "price": 0.1, "software": 0.15, "design": 0.5}
        neg_aspects = {"price": 0.9, "software": 0.8, "customer service": 0.7,
                       "battery life": 0.1, "performance": 0.15, "display": 0.2}
        reviews = [
            self._make_review("Positive", pos_aspects),
            self._make_review("Positive", pos_aspects),
            self._make_review("Negative", neg_aspects),
        ]
        pos, neg = _classify_aspects(reviews)
        assert "battery life" in pos
        assert "price" in neg

    def test_no_overlap_between_pos_and_neg(self):
        reviews = [
            self._make_review("Positive", {"battery life": 0.9, "design": 0.8}),
            self._make_review("Negative", {"price": 0.9, "software": 0.8}),
        ]
        pos, neg = _classify_aspects(reviews)
        for aspect in pos:
            assert aspect not in neg

    def test_all_positive_reviews_falls_back(self):
        """When all reviews are positive, neg list should use lowest-scored aspects."""
        aspects = {"battery life": 0.9, "performance": 0.8, "display": 0.7,
                   "price": 0.1, "software": 0.15, "design": 0.5}
        reviews = [
            self._make_review("Positive", aspects),
            self._make_review("Positive", aspects),
        ]
        pos, neg = _classify_aspects(reviews)
        assert len(pos) > 0
        assert len(neg) > 0
        # No overlap
        for a in neg:
            assert a not in pos

    def test_empty_reviews_returns_empty_lists(self):
        pos, neg = _classify_aspects([])
        assert pos == []
        assert neg == []

    def test_max_top_n_returned(self):
        aspects = {f"aspect_{i}": float(i) / 10 for i in range(10)}
        reviews = [self._make_review("Positive", aspects)]
        pos, neg = _classify_aspects(reviews)
        assert len(pos) <= 3
        assert len(neg) <= 3


# ---------------------------------------------------------------------------
# _determine_overall_sentiment
# ---------------------------------------------------------------------------

class TestDetermineOverallSentiment:
    def test_all_positive(self):
        label, score = _determine_overall_sentiment(10, 0)
        assert label == "Positive"
        assert score == 1.0

    def test_all_negative(self):
        label, score = _determine_overall_sentiment(0, 10)
        assert label == "Negative"

    def test_mixed_equal(self):
        label, score = _determine_overall_sentiment(5, 5)
        assert label == "Mixed"

    def test_mostly_positive(self):
        label, score = _determine_overall_sentiment(8, 2)
        assert label == "Positive"
        assert score > 0.5

    def test_mostly_negative(self):
        label, score = _determine_overall_sentiment(2, 8)
        assert label == "Negative"

    def test_zero_total_returns_mixed(self):
        label, score = _determine_overall_sentiment(0, 0)
        assert label == "Mixed"
        assert score == 0.5

    def test_score_in_range(self):
        for pos, neg in [(7, 3), (3, 7), (5, 5), (10, 0), (0, 10)]:
            _, score = _determine_overall_sentiment(pos, neg)
            assert 0.0 <= score <= 1.0, f"Score out of range for ({pos}, {neg}): {score}"


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_contains_review_count(self):
        s = _build_summary(10, "smartphone", "Positive", 0.8, ["battery life"], ["price"])
        assert "10" in s

    def test_contains_category(self):
        s = _build_summary(5, "laptop", "Positive", 0.9, [], [])
        assert "laptop" in s

    def test_contains_sentiment(self):
        s = _build_summary(5, "smartphone", "Negative", 0.7, [], [])
        assert "Negative" in s

    def test_contains_positive_aspect(self):
        s = _build_summary(3, "smartphone", "Mixed", 0.6, ["battery life"], [])
        assert "battery life" in s

    def test_contains_negative_aspect(self):
        s = _build_summary(3, "smartphone", "Mixed", 0.6, [], ["price"])
        assert "price" in s

    def test_empty_aspects_use_fallback(self):
        s = _build_summary(1, "smartphone", "Positive", 0.9, [], [])
        assert "overall quality" in s or "minor issues" in s

    def test_singular_review(self):
        s = _build_summary(1, "smartphone", "Positive", 0.9, [], [])
        assert "1 review" in s
        assert "reviews" not in s

    def test_plural_reviews(self):
        s = _build_summary(5, "smartphone", "Positive", 0.9, [], [])
        assert "5 reviews" in s


# ---------------------------------------------------------------------------
# _keyword_category
# ---------------------------------------------------------------------------

class TestKeywordCategory:
    def test_phone_keyword_detected(self):
        text = "This phone has great battery life and a stunning camera."
        result = _keyword_category(text)
        assert result == "smartphone"

    def test_iphone_detected(self):
        assert _keyword_category("The iphone is amazing") == "smartphone"

    def test_android_detected(self):
        assert _keyword_category("My android phone is fast") == "smartphone"

    def test_laptop_detected(self):
        assert _keyword_category("This laptop is great for work") == "laptop"

    def test_headphone_detected(self):
        assert _keyword_category("These earbuds have great sound") == "headphones"

    def test_no_match_returns_none(self):
        result = _keyword_category("This product is amazing and worth the price.")
        assert result is None

    def test_camera_feature_word_not_classified_as_camera(self):
        # "camera quality" should NOT trigger the camera category
        text = "The camera quality on this phone is stunning"
        result = _keyword_category(text)
        # Should detect "phone" → smartphone, not camera
        assert result == "smartphone"

    def test_empty_returns_none(self):
        assert _keyword_category("") is None

    def test_case_insensitive(self):
        assert _keyword_category("This PHONE is great") == "smartphone"


# ---------------------------------------------------------------------------
# analyze_reviews — full integration with mocked models
# ---------------------------------------------------------------------------

# Reusable mock return values
_POSITIVE_SENTIMENT = {"label": "POSITIVE", "score": 0.95, "emoji": "😊"}
_NEGATIVE_SENTIMENT = {"label": "NEGATIVE", "score": 0.88, "emoji": "😞"}
_MOCK_ASPECTS = {
    "battery life": 0.5, "camera quality": 0.3, "display": 0.2,
    "performance": 0.4, "price": 0.15, "design": 0.25,
    "sound quality": 0.1, "build quality": 0.2, "software": 0.1,
    "customer service": 0.05,
}


@patch("analyzer.predict_sentiment")
@patch("analyzer.predict_aspect")
@patch("analyzer.predict_category", return_value="smartphone")
class TestAnalyzeReviews:

    def test_returns_dict_with_required_keys(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Great phone with amazing battery life."])

        required_keys = [
            "overall_sentiment", "sentiment_score", "positive_count",
            "negative_count", "product_category", "top_positive_aspects",
            "top_negative_aspects", "most_discussed_topics", "summary",
            "per_review_results", "aggregated_aspects",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_all_positive_reviews(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Great battery.", "Amazing camera.", "Love the design."])

        assert result["positive_count"] == 3
        assert result["negative_count"] == 0
        assert result["overall_sentiment"] == "Positive"

    def test_all_negative_reviews(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _NEGATIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Terrible battery.", "Bad camera.", "Poor design."])

        assert result["negative_count"] == 3
        assert result["positive_count"] == 0
        assert result["overall_sentiment"] == "Negative"

    def test_mixed_reviews(self, mock_cat, mock_asp, mock_sent):
        sentiments = [_POSITIVE_SENTIMENT, _NEGATIVE_SENTIMENT,
                      _POSITIVE_SENTIMENT, _NEGATIVE_SENTIMENT]
        mock_sent.side_effect = sentiments * 10  # enough for any call count
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Good.", "Bad.", "Good.", "Bad."])

        assert result["overall_sentiment"] == "Mixed"

    def test_per_review_results_count(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        reviews = ["Review one.", "Review two.", "Review three."]
        result = analyze_reviews(reviews)

        assert len(result["per_review_results"]) == 3

    def test_per_review_has_required_fields(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Good product."])
        r = result["per_review_results"][0]

        assert "index" in r
        assert "review_snippet" in r
        assert "sentiment" in r
        assert "score" in r
        assert "emoji" in r

    def test_sentiment_score_in_range(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Good product."])

        assert 0.0 <= result["sentiment_score"] <= 1.0

    def test_most_discussed_topics_max_5(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews([
            "Battery, camera, screen, performance, price, sound, software mentioned."
        ])
        assert len(result["most_discussed_topics"]) <= 5

    def test_clamps_to_100_reviews(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        reviews = [f"Review number {i}." for i in range(150)]
        result = analyze_reviews(reviews)

        assert len(result["per_review_results"]) == 100

    def test_empty_input_raises(self, mock_cat, mock_asp, mock_sent):
        from analyzer import analyze_reviews
        with pytest.raises(ValueError):
            analyze_reviews([])

    def test_blank_lines_filtered(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["   ", "Good product.", "  "])

        assert len(result["per_review_results"]) == 1

    def test_category_uses_keyword_shortcut(self, mock_cat, mock_asp, mock_sent):
        """Keyword pre-check should resolve 'phone' without calling zero-shot."""
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS
        mock_cat.reset_mock()

        from analyzer import analyze_reviews
        result = analyze_reviews(["This phone has great battery life."])

        # keyword matched 'smartphone' → zero-shot should NOT have been called
        mock_cat.assert_not_called()
        assert result["product_category"] == "smartphone"

    def test_summary_is_non_empty_string(self, mock_cat, mock_asp, mock_sent):
        mock_sent.return_value = _POSITIVE_SENTIMENT
        mock_asp.return_value = _MOCK_ASPECTS

        from analyzer import analyze_reviews
        result = analyze_reviews(["Good phone."])

        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 20
