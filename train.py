"""
train.py — Fine-tune DistilBERT on amazon_polarity for sentiment classification.

What this script does
---------------------
1. Loads a stratified subset of amazon_polarity (train split)
2. Tokenizes reviews with DistilBertTokenizerFast
3. Fine-tunes distilbert-base-uncased for 3 epochs using HuggingFace Trainer
4. Records per-epoch train loss, eval loss, and eval accuracy
5. Saves the fine-tuned model to ./fine_tuned_model/
6. Generates and saves two figures to report_figures/:
     - training_curves.png  (loss + accuracy over epochs)
     - model_comparison.png (baseline vs fine-tuned metrics)

Run:
    python train.py

Expected runtime: ~8–15 min on CPU (1000 train + 300 eval samples).
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "fine_tuned_model")
FIGURES_DIR = os.path.join(BASE_DIR, "report_figures")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.json")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL     = "distilbert-base-uncased"
TRAIN_SAMPLES  = 1000   # increase for better accuracy; 1000 is fast on CPU
EVAL_SAMPLES   = 300
NUM_EPOCHS     = 3
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
MAX_LENGTH     = 128    # truncate tokens (faster than 512 for CPU)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})

# ---------------------------------------------------------------------------
# Callback to collect per-epoch metrics
# ---------------------------------------------------------------------------

class MetricsLogger(TrainerCallback):
    """Stores train/eval loss and eval accuracy after each epoch."""

    def __init__(self):
        self.train_loss:  list[float] = []
        self.eval_loss:   list[float] = []
        self.eval_acc:    list[float] = []
        self._last_train_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:                      # training step log
            self._last_train_loss = logs["loss"]
        if "eval_loss" in logs:                 # end-of-epoch eval log
            self.eval_loss.append(logs["eval_loss"])
            if self._last_train_loss is not None:
                self.train_loss.append(self._last_train_loss)
            if "eval_accuracy" in logs:
                self.eval_acc.append(logs["eval_accuracy"])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_splits(train_n: int, eval_n: int):
    """Load stratified subsets from amazon_polarity."""
    print(f"Loading dataset  (train={train_n}, eval={eval_n})…")
    train_raw = load_dataset("amazon_polarity", split=f"train[:{train_n}]")
    eval_raw  = load_dataset("amazon_polarity", split=f"test[:{eval_n}]")
    return train_raw, eval_raw


def tokenize_dataset(raw_ds, tokenizer, max_length: int):
    """Tokenize and return a HuggingFace Dataset ready for Trainer."""
    def _tokenize(batch):
        return tokenizer(
            batch["content"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = raw_ds.map(_tokenize, batched=True, remove_columns=["title", "content"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


# ---------------------------------------------------------------------------
# Metrics function (called by Trainer after each eval)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall":    recall_score(labels, preds, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Figure: training curves
# ---------------------------------------------------------------------------

def plot_training_curves(logger: MetricsLogger, num_epochs: int) -> None:
    epochs = list(range(1, len(logger.eval_loss) + 1))
    if not epochs:
        print("Warning: no epoch-level metrics captured — skipping training curves plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    if logger.train_loss:
        ax1.plot(epochs[:len(logger.train_loss)], logger.train_loss,
                 marker="o", label="Train Loss", color="#6366f1")
    ax1.plot(epochs, logger.eval_loss,
             marker="s", label="Eval Loss", color="#f59e0b")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(epochs)
    ax1.legend()

    # Accuracy curve
    if logger.eval_acc:
        ax2.plot(epochs[:len(logger.eval_acc)], logger.eval_acc,
                 marker="o", color="#22c55e", linewidth=2)
        ax2.set_ylim(0, 1)
        for x, y in zip(epochs[:len(logger.eval_acc)], logger.eval_acc):
            ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
    ax2.set_title("Eval Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(epochs)

    fig.suptitle(
        f"DistilBERT Fine-tuning on amazon_polarity  "
        f"(n_train={TRAIN_SAMPLES}, epochs={NUM_EPOCHS}, lr={LEARNING_RATE})",
        fontsize=11,
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "training_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure: baseline vs fine-tuned comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(baseline: dict, finetuned: dict) -> None:
    metric_names = ["accuracy", "f1", "precision", "recall"]
    labels       = ["Accuracy", "F1", "Precision", "Recall"]
    x     = np.arange(len(metric_names))
    width = 0.35

    baseline_vals  = [baseline.get(m, 0)  for m in metric_names]
    finetuned_vals = [finetuned.get(m, 0) for m in metric_names]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width / 2, baseline_vals,  width, label="Baseline (pre-trained)",
                   color="#94a3b8", alpha=0.85)
    bars2 = ax.bar(x + width / 2, finetuned_vals, width, label="Fine-tuned",
                   color="#6366f1", alpha=0.85)

    for bar, val in [*zip(bars1, baseline_vals), *zip(bars2, finetuned_vals)]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Baseline vs Fine-tuned DistilBERT — amazon_polarity (n=300)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Evaluate fine-tuned model on test set (manual loop, no pipeline)
# ---------------------------------------------------------------------------

def evaluate_finetuned(trainer: Trainer, eval_dataset) -> dict:
    """Run Trainer.evaluate() and return clean metrics dict."""
    results = trainer.evaluate(eval_dataset=eval_dataset)
    return {
        "accuracy":  round(results.get("eval_accuracy",  0), 4),
        "f1":        round(results.get("eval_f1",        0), 4),
        "precision": round(results.get("eval_precision", 0), 4),
        "recall":    round(results.get("eval_recall",    0), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Product Insight AI — Fine-tuning DistilBERT")
    print("=" * 60)
    t_start = time.time()

    # 1. Load data
    train_raw, eval_raw = load_splits(TRAIN_SAMPLES, EVAL_SAMPLES)

    # 2. Tokenizer
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    # 3. Tokenize
    print("Tokenizing…")
    train_ds = tokenize_dataset(train_raw, tokenizer, MAX_LENGTH)
    eval_ds  = tokenize_dataset(eval_raw,  tokenizer, MAX_LENGTH)

    # 4. Model
    print(f"Loading model: {BASE_MODEL}")
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",          # disable wandb / tensorboard
        use_cpu=True,              # CPU training — set False if GPU available
    )

    # 6. Callback + Trainer
    logger = MetricsLogger()
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[logger],
    )

    # 7. Train
    print(f"\nStarting fine-tuning  "
          f"(epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE})…\n")
    trainer.train()

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min.")

    # 8. Save fine-tuned model + tokenizer
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")

    # 9. Evaluate fine-tuned
    print("\nEvaluating fine-tuned model…")
    ft_metrics = evaluate_finetuned(trainer, eval_ds)
    print(f"  Fine-tuned  →  Acc={ft_metrics['accuracy']:.4f}  "
          f"F1={ft_metrics['f1']:.4f}  "
          f"Prec={ft_metrics['precision']:.4f}  "
          f"Rec={ft_metrics['recall']:.4f}")

    # 10. Load baseline metrics (from evaluate.py, if available)
    baseline_metrics = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            baseline_metrics = json.load(f)
        print(f"  Baseline    →  Acc={baseline_metrics.get('accuracy',0):.4f}  "
              f"F1={baseline_metrics.get('f1',0):.4f}")
    else:
        print("  Baseline metrics not found — run evaluate.py first for comparison.")

    # 11. Save fine-tuned metrics
    ft_metrics["model"] = "fine-tuned"
    ft_metrics["n_train"] = TRAIN_SAMPLES
    ft_metrics["n_eval"]  = EVAL_SAMPLES
    ft_metrics["epochs"]  = NUM_EPOCHS
    ft_metrics_path = os.path.join(BASE_DIR, "metrics_finetuned.json")
    with open(ft_metrics_path, "w") as f:
        json.dump(ft_metrics, f, indent=2)
    print(f"Saved: {ft_metrics_path}")

    # 12. Figures
    print("\nGenerating figures…")
    plot_training_curves(logger, NUM_EPOCHS)
    if baseline_metrics:
        plot_model_comparison(baseline_metrics, ft_metrics)
    else:
        print("Skipping model_comparison.png (no baseline metrics).")

    print("\nDone. Run 'python app.py' to use the fine-tuned model in the dashboard.")


if __name__ == "__main__":
    main()
