"""
app.py — Gradio dashboard for Product Insight AI.

Tabs
----
  1. Analyze Product  — paste reviews, get full insight dashboard
  2. Batch File       — upload .txt / .csv file, analyze all reviews
  3. History          — auto-saved session log with click-to-review
  4. Model Performance— metrics.json + saved report figures

Launch:
    python app.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime

import gradio as gr
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import analyze_reviews
from utils import (
    format_score,
    render_aspect_bars,
    render_aspect_columns,
    render_sentiment_badge,
    reviews_from_text,
    t,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGURES      = os.path.join(BASE_DIR, "report_figures")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Example reviews
# ---------------------------------------------------------------------------

EXAMPLE_REVIEWS = """The battery life on this iPhone is absolutely amazing, easily lasts two full days with heavy use.
The camera is stunning. Night mode photos look professional and the video stabilization is flawless.
Best smartphone I have ever owned. The display is vivid and the performance is buttery smooth.
The price is way too high for what you get. My previous Android had the same specs for half the cost.
The charging speed is disappointingly slow compared to competitors. No fast charger in the box either.
After the latest iOS update the phone keeps crashing and apps freeze randomly. Very frustrating."""

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

_history: list[dict] = []


def _save_to_history(result: dict) -> None:
    _history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_reviews": result["positive_count"] + result["negative_count"],
        "category":  result["product_category"],
        "sentiment": result["overall_sentiment"],
        "score":     format_score(result["sentiment_score"]),
        "summary":   result["summary"],
    })


def _history_dataframe() -> pd.DataFrame:
    if not _history:
        return pd.DataFrame(columns=["Timestamp", "# Reviews", "Category", "Sentiment", "Score"])
    df = pd.DataFrame(_history)
    df.columns = ["Timestamp", "# Reviews", "Category", "Sentiment", "Score", "summary"]
    return df.drop(columns=["summary"])


# ---------------------------------------------------------------------------
# Core analysis wrapper
# ---------------------------------------------------------------------------

def run_analysis(raw_text: str):
    """Parse text → analyze → return all Gradio outputs."""
    reviews = reviews_from_text(raw_text)
    if not reviews:
        err = "<p style='color:#ef4444;font-weight:600'>Please enter at least one review.</p>"
        return err, "", "", "", "", pd.DataFrame(), "No reviews entered."

    try:
        result = analyze_reviews(reviews)
    except Exception:
        err = f"<pre style='color:red'>{traceback.format_exc()}</pre>"
        return err, "", "", "", "", pd.DataFrame(), "Error during analysis."

    _save_to_history(result)

    # Sentiment badge
    badge = render_sentiment_badge(result["overall_sentiment"], result["sentiment_score"])

    # Category card
    cat_icon = {
        "smartphone": "📱", "laptop": "💻", "headphones": "🎧",
        "tablet": "📲", "smartwatch": "⌚", "camera": "📷",
        "TV": "📺", "other": "📦",
    }.get(result["product_category"], "📦")

    cat_html = f"""
    <div style="padding:16px;background:#f0f9ff;border:1px solid #bae6fd;
                border-radius:12px;text-align:center;">
      <div style="font-size:32px">{cat_icon}</div>
      <div style="font-size:18px;font-weight:700;color:#0369a1;margin-top:6px;">
        {result['product_category'].title()}
      </div>
      <div style="font-size:12px;color:#64748b;">Detected Category</div>
    </div>"""

    # Three-column aspects / topics
    columns_html = render_aspect_columns(
        result["top_positive_aspects"],
        result["top_negative_aspects"],
        result["most_discussed_topics"],
    )

    # Summary
    summary_html = f"""
    <div style="padding:16px;background:#fafafa;border-left:4px solid #6366f1;
                border-radius:8px;font-size:15px;line-height:1.7;color:#1f2937;">
      {result['summary']}
    </div>"""

    # Aspect bars
    aspects_html = render_aspect_bars(result.get("aggregated_aspects", {}))

    # Per-review table
    per_df = pd.DataFrame([
        {
            "#": r["index"],
            "Review": r["review_snippet"],
            "Sentiment": f"{r['emoji']} {r['sentiment']}",
            "Confidence": format_score(r["score"]),
        }
        for r in result["per_review_results"]
    ])

    status = (
        f"✅ {len(reviews)} review{'s' if len(reviews) != 1 else ''} analyzed — "
        f"{result['overall_sentiment']} ({format_score(result['sentiment_score'])})"
    )

    return badge, cat_html, columns_html, summary_html, aspects_html, per_df, status


# ---------------------------------------------------------------------------
# File upload handler
# ---------------------------------------------------------------------------

def handle_file_upload(file_obj):
    if file_obj is None:
        return ""
    try:
        filepath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext == ".csv":
            df = pd.read_csv(filepath)
            col = next(
                (c for c in ("review", "text", "content", "Review", "Text", "Content")
                 if c in df.columns),
                df.columns[0],
            )
            return "\n".join(df[col].dropna().astype(str).tolist())
        return ""
    except Exception as exc:
        return f"Error reading file: {exc}"


# ---------------------------------------------------------------------------
# CSV download
# ---------------------------------------------------------------------------

def build_download_csv(raw_text: str):
    reviews = reviews_from_text(raw_text)
    if not reviews:
        return None
    try:
        result = analyze_reviews(reviews)
    except Exception:
        return None

    rows = [
        {"index": r["index"], "review": r["review_snippet"],
         "sentiment": r["sentiment"], "confidence": r["score"]}
        for r in result["per_review_results"]
    ]
    rows.append({"index": "", "review": "OVERALL",
                 "sentiment": result["overall_sentiment"],
                 "confidence": result["sentiment_score"]})

    out_path = os.path.join(BASE_DIR, "_download_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Performance tab
# ---------------------------------------------------------------------------

def load_metrics_html() -> str:
    if not os.path.exists(METRICS_FILE):
        return "<p style='color:#ef4444'>metrics.json not found. Run evaluate.py first.</p>"

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        m = json.load(f)

    def card(label, val, colour):
        return (
            f"<div style='flex:1;padding:20px;background:{colour}1a;border:2px solid {colour};"
            f"border-radius:14px;text-align:center;min-width:120px'>"
            f"<div style='font-size:28px;font-weight:800;color:{colour}'>{val:.1%}</div>"
            f"<div style='font-size:13px;color:#6b7280;margin-top:4px'>{label}</div></div>"
        )

    cards = "".join([
        card("Accuracy",  m["accuracy"],  "#6366f1"),
        card("F1 Score",  m["f1"],        "#22c55e"),
        card("Precision", m["precision"], "#f59e0b"),
        card("Recall",    m["recall"],    "#3b82f6"),
    ])
    return f"<div style='display:flex;gap:12px;flex-wrap:wrap;justify-content:center'>{cards}</div>"


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Product Insight AI 🔍") as demo:

        # Header
        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px">
          <h1 style="font-size:2.2rem;font-weight:800;color:#1e3a5f;margin:0">
            🔍 Product Insight AI
          </h1>
          <p style="color:#64748b;font-size:1rem;margin-top:6px">
            Smart Amazon Review Analyzer — powered by DistilBERT &amp; BART
          </p>
        </div>""")

        with gr.Tabs():

            # ==============================================================
            # TAB 1 — Analyze Product
            # ==============================================================
            with gr.Tab("🔍 Analyze Product"):

                review_input = gr.Textbox(
                    lines=10,
                    placeholder=t("input_placeholder"),
                    label="Reviews (one per line)",
                )

                with gr.Row():
                    btn_example = gr.Button("📋 Load Example", variant="secondary")
                    btn_analyze = gr.Button("🚀 Analyze",       variant="primary")
                    btn_clear   = gr.Button("🗑️ Clear",         variant="stop")

                status_out = gr.Textbox(label="Status", interactive=False, max_lines=2)

                with gr.Row():
                    badge_out    = gr.HTML(label="Sentiment")
                    category_out = gr.HTML(label="Category")

                columns_out    = gr.HTML(label="Aspects & Topics")
                summary_out    = gr.HTML(label="Summary")
                aspects_out    = gr.HTML(label="Aspect Scores")
                per_review_out = gr.Dataframe(label="Per-Review Breakdown", wrap=True)

                btn_example.click(lambda: EXAMPLE_REVIEWS, outputs=review_input)

                btn_clear.click(
                    lambda: ("", "", "", "", "", "", pd.DataFrame(), ""),
                    outputs=[review_input, badge_out, category_out,
                             columns_out, summary_out, aspects_out,
                             per_review_out, status_out],
                )

                btn_analyze.click(
                    run_analysis,
                    inputs=review_input,
                    outputs=[badge_out, category_out, columns_out,
                             summary_out, aspects_out, per_review_out, status_out],
                )

            # ==============================================================
            # TAB 2 — Batch File Analysis
            # ==============================================================
            with gr.Tab("📁 Batch File Analysis"):

                file_input = gr.File(
                    label="Upload .txt or .csv",
                    file_types=[".txt", ".csv"],
                )

                with gr.Row():
                    btn_load_file   = gr.Button("📂 Load File",    variant="secondary")
                    btn_analyze_all = gr.Button("🚀 Analyze All",  variant="primary")

                batch_text = gr.Textbox(
                    lines=8,
                    label="Loaded Reviews (editable)",
                    placeholder="Reviews will appear here after loading…",
                )

                status_batch   = gr.Textbox(label="Status", interactive=False, max_lines=2)

                with gr.Row():
                    badge_batch    = gr.HTML()
                    category_batch = gr.HTML()

                columns_batch = gr.HTML()
                summary_batch = gr.HTML()
                aspects_batch = gr.HTML()
                per_batch_df  = gr.Dataframe(label="Per-Review Breakdown", wrap=True)

                with gr.Row():
                    download_btn  = gr.Button("⬇️ Download Results as CSV", variant="secondary")
                    download_file = gr.File(label="Download")

                btn_load_file.click(handle_file_upload, inputs=file_input, outputs=batch_text)

                btn_analyze_all.click(
                    run_analysis,
                    inputs=batch_text,
                    outputs=[badge_batch, category_batch, columns_batch,
                             summary_batch, aspects_batch, per_batch_df, status_batch],
                )

                download_btn.click(build_download_csv, inputs=batch_text, outputs=download_file)

            # ==============================================================
            # TAB 3 — History
            # ==============================================================
            with gr.Tab("🕓 History"):

                with gr.Row():
                    btn_refresh      = gr.Button("🔄 Refresh")
                    btn_clear_hist   = gr.Button("🗑️ Clear History", variant="stop")

                history_table    = gr.Dataframe(label="Analysis History", wrap=True, interactive=False)
                selected_summary = gr.Textbox(label="Selected Session Summary", lines=4, interactive=False)

                btn_refresh.click(lambda: _history_dataframe(), outputs=history_table)
                btn_clear_hist.click(
                    lambda: (_history.clear() or pd.DataFrame(), ""),
                    outputs=[history_table, selected_summary],
                )

                def on_select(evt: gr.SelectData):
                    row_idx = evt.index[0]
                    if 0 <= row_idx < len(_history):
                        return _history[row_idx].get("summary", "")
                    return ""

                history_table.select(on_select, outputs=selected_summary)

            # ==============================================================
            # TAB 4 — Model Performance
            # ==============================================================
            with gr.Tab("📊 Model Performance"):

                btn_load_metrics = gr.Button("📈 Load Metrics")
                metrics_html     = gr.HTML()
                explanation      = gr.HTML(
                    f"<p style='color:#6b7280;font-size:14px;padding:8px 0'>{t('perf_explanation')}</p>"
                )

                btn_load_metrics.click(load_metrics_html, outputs=metrics_html)

                gr.HTML("<hr style='margin:16px 0;border-color:#e5e7eb'><h3 style='color:#1e3a5f'>Figures</h3>")

                # Which model is active?
                from model import USE_FINETUNED, SENTIMENT_MODEL
                model_badge = (
                    "<span style='background:#6366f1;color:white;padding:4px 10px;"
                    "border-radius:20px;font-size:13px;font-weight:600'>"
                    "🔬 Fine-tuned model active</span>"
                    if USE_FINETUNED else
                    "<span style='background:#94a3b8;color:white;padding:4px 10px;"
                    "border-radius:20px;font-size:13px;font-weight:600'>"
                    "📦 Baseline model active (run train.py to fine-tune)</span>"
                )
                gr.HTML(f"<div style='margin-bottom:12px'>{model_badge}</div>")

                # Row 1: evaluation figures
                fig_paths = {
                    "Confusion Matrix":   os.path.join(FIGURES, "confusion_matrix.png"),
                    "Metrics by Class":   os.path.join(FIGURES, "metrics_bar.png"),
                    "Topic Distribution": os.path.join(FIGURES, "topic_distribution.png"),
                    "Sample Predictions": os.path.join(FIGURES, "sample_predictions.png"),
                }

                with gr.Row():
                    for label, path in fig_paths.items():
                        if os.path.exists(path):
                            gr.Image(value=path, label=label)
                        else:
                            gr.HTML(
                                f"<div style='padding:40px;text-align:center;color:#9ca3af;"
                                f"border:1px dashed #d1d5db;border-radius:8px'>"
                                f"📊 {label}<br><small>Run evaluate.py to generate</small></div>"
                            )

                # Row 2: training curves + model comparison (only after train.py)
                train_figs = {
                    "Training Curves":    os.path.join(FIGURES, "training_curves.png"),
                    "Model Comparison":   os.path.join(FIGURES, "model_comparison.png"),
                }
                if any(os.path.exists(p) for p in train_figs.values()):
                    gr.HTML("<hr style='margin:16px 0'><h3 style='color:#1e3a5f'>Fine-tuning Results</h3>")
                    with gr.Row():
                        for label, path in train_figs.items():
                            if os.path.exists(path):
                                gr.Image(value=path, label=label)

        gr.HTML("""
        <div style="text-align:center;padding:12px;color:#9ca3af;font-size:12px">
          Product Insight AI · DistilBERT + BART-MNLI
        </div>""")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    )
