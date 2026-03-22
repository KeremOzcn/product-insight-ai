"""
evaluate.py — Offline evaluation of the sentiment model on amazon_polarity.

Loads 300 test samples, runs predict_sentiment(), computes classification
metrics, and saves four figures to report_figures/:

  1. confusion_matrix.png
  2. metrics_bar.png
  3. sample_predictions.png
  4. topic_distribution.png

Also writes metrics.json for the Gradio performance tab.

Run:
    python evaluate.py
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from analyzer import TOPIC_KEYWORDS
from model import predict_sentiment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "report_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Label mapping
# amazon_polarity: label 0 = negative, label 1 = positive
# model output:    "NEGATIVE" / "POSITIVE"
# ---------------------------------------------------------------------------

_INT_TO_STR = {0: "NEGATIVE", 1: "POSITIVE"}
_STR_TO_INT = {"NEGATIVE": 0, "POSITIVE": 1}

# Matplotlib / Seaborn style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(n: int = 300) -> pd.DataFrame:
    """Load *n* test samples from amazon_polarity and return a DataFrame."""
    print(f"Loading {n} test samples from amazon_polarity…")
    dataset = load_dataset("amazon_polarity", split=f"test[:{n}]")
    df = pd.DataFrame({"text": dataset["content"], "true_int": dataset["label"]})
    df["true_label"] = df["true_int"].map(_INT_TO_STR)
    return df


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Add pred_label and pred_score columns to *df* in-place and return it."""
    print(f"Running inference on {len(df)} samples…")
    t0 = time.time()

    pred_labels: list[str] = []
    pred_scores: list[float] = []

    for i, text in enumerate(df["text"], 1):
        result = predict_sentiment(text)
        pred_labels.append(result["label"])
        pred_scores.append(result["score"])

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(df)} done  ({elapsed:.1f}s elapsed)")

    df["pred_label"] = pred_labels
    df["pred_score"] = pred_scores
    df["pred_int"] = df["pred_label"].map(_STR_TO_INT)

    total_time = time.time() - t0
    print(f"Inference complete in {total_time:.1f}s ({total_time/len(df):.2f}s/sample).")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute and return a metrics dict."""
    y_true = df["true_int"].tolist()
    y_pred = df["pred_int"].tolist()

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="weighted")
    prec  = precision_score(y_true, y_pred, average="weighted")
    rec   = recall_score(y_true, y_pred, average="weighted")

    report = classification_report(
        y_true, y_pred,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )

    metrics = {
        "accuracy":  round(acc,  4),
        "f1":        round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "per_class": {
            "Negative": {k: round(v, 4) for k, v in report["Negative"].items()},
            "Positive": {k: round(v, 4) for k, v in report["Positive"].items()},
        },
        "n_samples": len(df),
    }
    print(
        f"\nMetrics (n={len(df)})\n"
        f"  Accuracy : {acc:.4f}\n"
        f"  F1       : {f1:.4f}\n"
        f"  Precision: {prec:.4f}\n"
        f"  Recall   : {rec:.4f}\n"
    )
    return metrics


# ---------------------------------------------------------------------------
# Figure 1 — Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame) -> None:
    """Save a seaborn confusion-matrix heatmap."""
    y_true = df["true_int"].tolist()
    y_pred = df["pred_int"].tolist()

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix — Sentiment Model", pad=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Per-class metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_bar(metrics: dict) -> None:
    """Save a grouped bar chart of precision / recall / F1 per class."""
    classes = ["Negative", "Positive"]
    metric_names = ["precision", "recall", "f1-score"]
    colours = ["#3b82f6", "#22c55e", "#f59e0b"]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, colour) in enumerate(zip(metric_names, colours)):
        values = [metrics["per_class"][c][metric] for c in classes]
        bars = ax.bar(x + i * width, values, width, label=metric.title(), color=colour, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_title("Per-Class Metrics — Sentiment Model", pad=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend()

    # Overall metrics as annotation
    overall_text = (
        f"Overall  Acc={metrics['accuracy']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"Prec={metrics['precision']:.3f}  "
        f"Rec={metrics['recall']:.3f}"
    )
    ax.text(
        0.5, -0.13, overall_text,
        transform=ax.transAxes, ha="center", fontsize=9, color="#6b7280",
    )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "metrics_bar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Sample predictions table
# ---------------------------------------------------------------------------

def plot_sample_predictions(df: pd.DataFrame, n: int = 10) -> None:
    """Save a table image of *n* random sample predictions."""
    sample = df.sample(n=min(n, len(df)), random_state=42).copy()
    sample["review"] = sample["text"].str[:60] + "…"
    sample["correct"] = (sample["true_int"] == sample["pred_int"])

    display_df = sample[["review", "true_label", "pred_label", "pred_score", "correct"]].copy()
    display_df.columns = ["Review (truncated)", "True", "Predicted", "Confidence", "Correct?"]
    display_df["Confidence"] = display_df["Confidence"].map(lambda s: f"{s:.1%}")
    display_df["Correct?"] = display_df["Correct?"].map(lambda b: "✓" if b else "✗")
    display_df = display_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(display_df.columns))))

    # Colour header
    for j in range(len(display_df.columns)):
        tbl[0, j].set_facecolor("#1e3a5f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colour correct / incorrect rows
    for i in range(1, len(display_df) + 1):
        correct_val = display_df.iloc[i - 1]["Correct?"]
        colour = "#d1fae5" if correct_val == "✓" else "#fee2e2"
        for j in range(len(display_df.columns)):
            tbl[i, j].set_facecolor(colour)

    ax.set_title("Sample Predictions (10 random)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "sample_predictions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Topic distribution bar chart
# ---------------------------------------------------------------------------

def plot_topic_distribution(df: pd.DataFrame) -> None:
    """Count topic keyword hits across all reviews and save a bar chart."""
    topic_counts: dict[str, int] = {topic: 0 for topic in TOPIC_KEYWORDS}

    for text in df["text"]:
        lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                topic_counts[topic] += 1

    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    topics, counts = zip(*sorted_topics)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("muted", len(topics))
    bars = ax.bar(topics, counts, color=palette, edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Topic Distribution Across 300 Amazon Reviews", pad=14)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Review Count")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "topic_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Fine-tuned model evaluation (optional, runs if fine_tuned_model/ exists)
# ---------------------------------------------------------------------------

FINETUNED_METRICS_PATH = os.path.join(BASE_DIR, "metrics_finetuned.json")
FINETUNED_MODEL_DIR    = os.path.join(BASE_DIR, "fine_tuned_model")


def run_finetuned_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Run inference with the fine-tuned model and add ft_pred_* columns."""
    from transformers import pipeline as hf_pipeline
    print("Running inference with fine-tuned model…")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=FINETUNED_MODEL_DIR,
        truncation=True,
        max_length=128,
    )
    ft_labels, ft_scores = [], []
    for text in df["text"]:
        result = pipe(text[:512], truncation=True)[0]
        ft_labels.append(result["label"])
        ft_scores.append(result["score"])

    df["ft_pred_label"] = ft_labels
    df["ft_pred_score"] = ft_scores
    df["ft_pred_int"]   = df["ft_pred_label"].map(_STR_TO_INT)
    return df


def plot_comparison(baseline: dict, finetuned: dict) -> None:
    """Grouped bar chart comparing baseline vs fine-tuned overall metrics."""
    metric_names  = ["accuracy", "f1", "precision", "recall"]
    display_names = ["Accuracy", "F1", "Precision", "Recall"]
    x = np.arange(len(metric_names))
    w = 0.35

    b_vals = [baseline.get(m, 0)  for m in metric_names]
    f_vals = [finetuned.get(m, 0) for m in metric_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, b_vals, w, label="Baseline (pre-trained SST-2)",
                   color="#94a3b8", alpha=0.85)
    bars2 = ax.bar(x + w/2, f_vals, w, label="Fine-tuned (amazon_polarity)",
                   color="#6366f1", alpha=0.85)

    for bar, val in [*zip(bars1, b_vals), *zip(bars2, f_vals)]:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Baseline vs Fine-tuned DistilBERT — 300 amazon_polarity test samples")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Product Insight AI — Model Evaluation")
    print("=" * 60)

    df = load_samples(300)
    df = run_inference(df)

    metrics = compute_metrics(df)

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {METRICS_PATH}")

    # Generate figures
    print("\nGenerating figures…")
    plot_confusion_matrix(df)
    plot_metrics_bar(metrics)
    plot_sample_predictions(df)
    plot_topic_distribution(df)

    # Optional: evaluate fine-tuned model and generate comparison figure
    ft_metrics = None
    if os.path.isdir(FINETUNED_MODEL_DIR) and os.path.exists(
        os.path.join(FINETUNED_MODEL_DIR, "config.json")
    ):
        print("\nFine-tuned model detected — running comparison evaluation…")
        df = run_finetuned_inference(df)
        ft_metrics = {
            "accuracy":  round(accuracy_score(df["true_int"], df["ft_pred_int"]), 4),
            "f1":        round(f1_score(df["true_int"], df["ft_pred_int"], average="weighted"), 4),
            "precision": round(precision_score(df["true_int"], df["ft_pred_int"], average="weighted"), 4),
            "recall":    round(recall_score(df["true_int"], df["ft_pred_int"], average="weighted"), 4),
        }
        print(f"  Fine-tuned → Acc={ft_metrics['accuracy']:.4f}  F1={ft_metrics['f1']:.4f}")
        with open(FINETUNED_METRICS_PATH, "w") as f:
            json.dump(ft_metrics, f, indent=2)
        plot_comparison(metrics, ft_metrics)
    else:
        print("\nNo fine-tuned model found — skipping comparison. Run train.py first.")

    print("\nEvaluation complete. All outputs saved to report_figures/")


if __name__ == "__main__":
    main()
