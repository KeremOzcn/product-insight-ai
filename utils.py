"""
utils.py — Shared helpers for Product Insight AI.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# UI label strings (English only)
# ---------------------------------------------------------------------------

LABELS: dict[str, str] = {
    "tab_analyze":        "Analyze Product",
    "tab_batch":          "Batch File Analysis",
    "tab_history":        "History",
    "tab_performance":    "Model Performance",
    "input_placeholder":  "Paste Amazon reviews here, one per line...",
    "btn_analyze":        "Analyze",
    "btn_load_example":   "Load Example",
    "btn_clear":          "Clear",
    "btn_download":       "Download CSV",
    "btn_clear_history":  "Clear History",
    "btn_analyze_all":    "Analyze All",
    "overall_sentiment":  "Overall Sentiment",
    "confidence":         "Confidence Score",
    "category":           "Product Category",
    "positive_aspects":   "Top Positive Aspects",
    "negative_aspects":   "Top Negative Aspects",
    "most_discussed":     "Most Discussed Topics",
    "summary":            "Summary Report",
    "per_review":         "Per-Review Breakdown",
    "positive":           "Positive",
    "negative":           "Negative",
    "mixed":              "Mixed",
    "hist_time":          "Timestamp",
    "hist_count":         "# Reviews",
    "hist_category":      "Category",
    "hist_sentiment":     "Sentiment",
    "hist_score":         "Score",
    "perf_accuracy":      "Accuracy",
    "perf_f1":            "F1 Score",
    "perf_precision":     "Precision",
    "perf_recall":        "Recall",
    "perf_explanation":   (
        "The model was evaluated on 300 Amazon reviews. "
        "Binary sentiment classification (Positive / Negative) was performed "
        "using a DistilBERT-based sentiment-analysis pipeline."
    ),
    "file_label":         "Review File (.txt or .csv)",
    "no_reviews":         "Please enter at least one review.",
    "no_metrics":         "metrics.json not found. Run evaluate.py first.",
    "no_history":         "No history records yet.",
    "processing":         "Processing…",
}


def t(key: str) -> str:
    """Return the UI string for *key*."""
    return LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_score(score: float) -> str:
    """Return a percentage string, e.g. 0.873 → '87.3%'."""
    return f"{score * 100:.1f}%"


def sentiment_color(label: str) -> str:
    """Return a hex colour for a sentiment label string."""
    mapping = {
        "POSITIVE": "#22c55e", "Positive": "#22c55e", "positive": "#22c55e",
        "NEGATIVE": "#ef4444", "Negative": "#ef4444", "negative": "#ef4444",
        "MIXED":    "#f59e0b", "Mixed":    "#f59e0b", "mixed":    "#f59e0b",
    }
    return mapping.get(label, "#6b7280")


def reviews_from_text(raw: str) -> list[str]:
    """Split *raw* into individual review strings (one per line)."""
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def render_aspect_bars(aspects: dict) -> str:
    """Render a dict of {aspect: score} as HTML coloured progress bars."""
    if not aspects:
        return "<p style='color:#9ca3af'>No aspect data.</p>"

    rows = []
    for aspect, score in sorted(aspects.items(), key=lambda x: x[1], reverse=True):
        pct = score * 100
        colour = "#22c55e" if pct >= 60 else ("#f59e0b" if pct >= 35 else "#ef4444")
        rows.append(f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;font-size:13px;
                      color:#374151;margin-bottom:2px;">
            <span>{aspect.title()}</span>
            <span style="font-weight:600">{pct:.1f}%</span>
          </div>
          <div style="background:#e5e7eb;border-radius:9999px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{colour};
                        border-radius:9999px;"></div>
          </div>
        </div>""")

    return f"<div style='padding:4px 0'>{''.join(rows)}</div>"


def render_sentiment_badge(label: str, score: float) -> str:
    """Render a large coloured sentiment badge card as HTML."""
    colour = sentiment_color(label)
    emoji  = {"Positive": "😊", "Negative": "😞", "Mixed": "😐"}.get(label, "😐")
    return f"""
    <div style="text-align:center;padding:24px 16px;background:{colour}1a;
                border:2px solid {colour};border-radius:16px;">
      <div style="font-size:48px;margin-bottom:8px;">{emoji}</div>
      <div style="font-size:28px;font-weight:800;color:{colour};">{label.upper()}</div>
      <div style="font-size:15px;color:#6b7280;margin-top:4px;">
        Confidence: <strong>{format_score(score)}</strong>
      </div>
    </div>"""


def render_topic_list(topics: list[tuple]) -> str:
    """Render a list of (topic, count) tuples as a styled HTML list."""
    if not topics:
        return "<p style='color:#9ca3af'>No topic data.</p>"
    medals = ["🥇", "🥈", "🥉", "🔥", "🔥"]
    items = [
        f"<li style='padding:6px 0;border-bottom:1px solid #f3f4f6;'>"
        f"{medals[min(i,4)]} <strong>{topic.title()}</strong> "
        f"<span style='color:#6b7280;font-size:13px;'>({count} mentions)</span></li>"
        for i, (topic, count) in enumerate(topics)
    ]
    return f"<ul style='list-style:none;padding:0;margin:0'>{''.join(items)}</ul>"


def render_aspect_columns(
    pos_aspects: list[str],
    neg_aspects: list[str],
    topics: list[tuple],
) -> str:
    """Render three-column layout: Positive | Negative | Most Discussed."""
    def _list(items: list[str], colour: str, icon: str) -> str:
        if not items:
            return "<p style='color:#9ca3af'>—</p>"
        lis = "".join(
            f"<li style='padding:5px 0;color:{colour};font-size:14px;'>"
            f"{icon} {item.title()}</li>"
            for item in items
        )
        return f"<ul style='list-style:none;padding:0;margin:0'>{lis}</ul>"

    col = "flex:1;padding:16px;background:#f9fafb;border-radius:12px;border:1px solid #e5e7eb;"
    h   = "font-size:15px;font-weight:700;margin-bottom:10px;color:#1f2937;"

    return f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <div style="{col}"><div style="{h}">{t('positive_aspects')}</div>{_list(pos_aspects,'#22c55e','✅')}</div>
      <div style="{col}"><div style="{h}">{t('negative_aspects')}</div>{_list(neg_aspects,'#ef4444','❌')}</div>
      <div style="{col}"><div style="{h}">{t('most_discussed')}</div>{render_topic_list(topics)}</div>
    </div>"""
