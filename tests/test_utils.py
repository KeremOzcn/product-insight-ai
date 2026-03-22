"""
tests/test_utils.py — Unit tests for utils.py helpers.

Run:
    pytest tests/test_utils.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils import (
    format_score,
    render_aspect_bars,
    render_sentiment_badge,
    render_topic_list,
    render_aspect_columns,
    reviews_from_text,
    sentiment_color,
    t,
)


# ---------------------------------------------------------------------------
# t() — label lookup
# ---------------------------------------------------------------------------

class TestT:
    def test_known_key_returns_string(self):
        result = t("btn_analyze")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_key_returns_key_itself(self):
        assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"

    def test_positive_label(self):
        assert t("positive") == "Positive"

    def test_negative_label(self):
        assert t("negative") == "Negative"

    def test_mixed_label(self):
        assert t("mixed") == "Mixed"


# ---------------------------------------------------------------------------
# format_score
# ---------------------------------------------------------------------------

class TestFormatScore:
    def test_round_number(self):
        assert format_score(1.0) == "100.0%"

    def test_zero(self):
        assert format_score(0.0) == "0.0%"

    def test_typical(self):
        assert format_score(0.873) == "87.3%"

    def test_half(self):
        assert format_score(0.5) == "50.0%"

    def test_small(self):
        assert format_score(0.001) == "0.1%"


# ---------------------------------------------------------------------------
# sentiment_color
# ---------------------------------------------------------------------------

class TestSentimentColor:
    def test_positive_uppercase(self):
        assert sentiment_color("POSITIVE") == "#22c55e"

    def test_positive_titlecase(self):
        assert sentiment_color("Positive") == "#22c55e"

    def test_negative_uppercase(self):
        assert sentiment_color("NEGATIVE") == "#ef4444"

    def test_negative_titlecase(self):
        assert sentiment_color("Negative") == "#ef4444"

    def test_mixed(self):
        assert sentiment_color("Mixed") == "#f59e0b"

    def test_unknown_returns_grey(self):
        assert sentiment_color("unknown") == "#6b7280"


# ---------------------------------------------------------------------------
# reviews_from_text
# ---------------------------------------------------------------------------

class TestReviewsFromText:
    def test_basic_split(self):
        raw = "Great product.\nBad battery.\nOkay design."
        result = reviews_from_text(raw)
        assert result == ["Great product.", "Bad battery.", "Okay design."]

    def test_empty_string_returns_empty_list(self):
        assert reviews_from_text("") == []

    def test_none_like_empty_returns_empty(self):
        assert reviews_from_text("   \n  \n  ") == []

    def test_strips_whitespace(self):
        raw = "  Hello world.  \n  Another review.  "
        result = reviews_from_text(raw)
        assert result == ["Hello world.", "Another review."]

    def test_filters_blank_lines(self):
        raw = "First review.\n\n\nSecond review."
        result = reviews_from_text(raw)
        assert len(result) == 2

    def test_single_review(self):
        assert reviews_from_text("Just one.") == ["Just one."]

    def test_windows_line_endings(self):
        raw = "Review one.\r\nReview two."
        result = reviews_from_text(raw)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# render_aspect_bars
# ---------------------------------------------------------------------------

class TestRenderAspectBars:
    def test_returns_html_string(self):
        html = render_aspect_bars({"battery life": 0.8, "price": 0.3})
        assert isinstance(html, str)
        assert "<div" in html

    def test_empty_dict_returns_no_data_msg(self):
        html = render_aspect_bars({})
        assert "No aspect data" in html

    def test_high_score_gets_green(self):
        html = render_aspect_bars({"battery life": 0.9})
        assert "#22c55e" in html

    def test_low_score_gets_red(self):
        html = render_aspect_bars({"price": 0.1})
        assert "#ef4444" in html

    def test_aspect_name_appears_in_output(self):
        html = render_aspect_bars({"battery life": 0.7})
        assert "Battery Life" in html


# ---------------------------------------------------------------------------
# render_sentiment_badge
# ---------------------------------------------------------------------------

class TestRenderSentimentBadge:
    def test_positive_badge_contains_green(self):
        html = render_sentiment_badge("Positive", 0.9)
        assert "#22c55e" in html

    def test_negative_badge_contains_red(self):
        html = render_sentiment_badge("Negative", 0.8)
        assert "#ef4444" in html

    def test_contains_confidence_score(self):
        html = render_sentiment_badge("Positive", 0.873)
        assert "87.3%" in html

    def test_contains_emoji(self):
        html = render_sentiment_badge("Positive", 0.9)
        assert "😊" in html

    def test_mixed_contains_orange(self):
        html = render_sentiment_badge("Mixed", 0.5)
        assert "#f59e0b" in html


# ---------------------------------------------------------------------------
# render_topic_list
# ---------------------------------------------------------------------------

class TestRenderTopicList:
    def test_returns_html(self):
        html = render_topic_list([("battery", 5), ("camera", 3)])
        assert "<li" in html

    def test_empty_returns_no_data_msg(self):
        html = render_topic_list([])
        assert "No topic data" in html

    def test_topic_name_appears(self):
        html = render_topic_list([("battery", 5)])
        assert "Battery" in html

    def test_count_appears(self):
        html = render_topic_list([("battery", 7)])
        assert "7" in html

    def test_gold_medal_for_first(self):
        html = render_topic_list([("battery", 5), ("price", 3)])
        assert "🥇" in html


# ---------------------------------------------------------------------------
# render_aspect_columns
# ---------------------------------------------------------------------------

class TestRenderAspectColumns:
    def test_returns_html(self):
        html = render_aspect_columns(
            ["battery life", "performance"],
            ["price", "software"],
            [("battery", 4), ("price", 3)],
        )
        assert "<div" in html

    def test_positive_aspects_present(self):
        html = render_aspect_columns(["battery life"], [], [])
        assert "Battery Life" in html

    def test_negative_aspects_present(self):
        html = render_aspect_columns([], ["price"], [])
        assert "Price" in html

    def test_empty_lists_show_dash(self):
        html = render_aspect_columns([], [], [])
        assert "—" in html
