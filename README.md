# Product Insight AI 🔍

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-4.29-orange?logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Smart Amazon Review Analyzer** — paste product reviews and instantly get
> sentiment scores, aspect analysis, topic frequencies, and a generated summary
> report, all inside a bilingual (TR/EN) interactive dashboard.

---

## What It Does

Given a set of Amazon product reviews, **Product Insight AI** will:

1. **Score overall sentiment** — Positive / Negative / Mixed with confidence
2. **Detect product category** — smartphone, laptop, headphones, and more
3. **Extract positive & negative features** — which aspects users love or hate
4. **Identify most-discussed topics** — battery, camera, price, performance…
5. **Generate a concise summary report** — 3–4 sentence structured overview
6. **Display everything in a bilingual Gradio dashboard** — TR / EN toggle

---

## Architecture

```
User Reviews (text input / .txt / .csv)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                       analyzer.py                         │
│                                                           │
│  ┌─────────────────┐    ┌──────────────────────────────┐  │
│  │  DistilBERT     │    │  BART-MNLI (Zero-Shot)       │  │
│  │  Sentiment      │    │  ┌────────────────────────┐  │  │
│  │  POSITIVE /     │    │  │ Category Detection     │  │  │
│  │  NEGATIVE       │    │  │ (smartphone, laptop…)  │  │  │
│  │  + confidence   │    │  ├────────────────────────┤  │  │
│  └─────────────────┘    │  │ Aspect Scoring         │  │  │
│                         │  │ (battery, camera, …)   │  │  │
│  ┌─────────────────┐    │  └────────────────────────┘  │  │
│  │  Keyword-based  │    └──────────────────────────────┘  │
│  │  Topic Counter  │                                       │
│  └─────────────────┘                                       │
│                                                           │
│  Summary Builder → 3-sentence structured report           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   Gradio Dashboard (4 tabs, TR/EN bilingual)
```

---

## Project Structure

```
product-insight-ai/
├── app.py           # Gradio UI — 4 tabs, bilingual
├── model.py         # HuggingFace pipeline wrappers (cached)
├── analyzer.py      # Multi-review analysis engine
├── evaluate.py      # Offline evaluation on amazon_polarity
├── utils.py         # TR/EN labels, HTML renderers, helpers
├── requirements.txt
├── README.md
└── report_figures/
    ├── confusion_matrix.png
    ├── metrics_bar.png
    ├── topic_distribution.png
    └── sample_predictions.png
```

---

## Installation

```bash
# 1. Clone / download the project
cd product-insight-ai

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will download ~1.5 GB of model weights from HuggingFace.
> Subsequent runs load from cache instantly.

---

## Usage

### Launch the dashboard

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### Run the offline evaluation

```bash
python evaluate.py
```

This loads 300 amazon_polarity test samples, runs inference, prints metrics,
and saves four figures to `report_figures/`.

### Quick smoke-test of the analyzer

```bash
python analyzer.py
```

---

## Features

| Feature | Details |
|---|---|
| Sentiment scoring | DistilBERT SST-2 — per review & overall |
| Product detection | BART-MNLI zero-shot, 8 categories |
| Aspect analysis | 10 default aspects, zero-shot scored |
| Topic frequency | Keyword-based, 8 topic buckets |
| Bilingual UI | English & Turkish, toggle mid-session |
| Batch file upload | .txt (one per line) or .csv |
| Session history | Auto-saved, click to re-view |
| CSV download | Export per-review results |
| Evaluation mode | Accuracy / F1 / Precision / Recall + 4 figures |

---

## Dashboard Tabs

1. **Analyze Product** — Paste reviews, load example, click Analyze
2. **Batch File Analysis** — Upload `.txt` / `.csv`, download results
3. **History** — Timestamped log of every analysis session
4. **Model Performance** — Metrics cards + confusion matrix + figures

---

## Roadmap (Future Development)

- [ ] Fine-tune DistilBERT on domain-specific Amazon review data
- [ ] Add multilingual review support (Turkish, German, French)
- [ ] Integrate with Amazon Product Advertising API for live review fetch
- [ ] Trend analysis — sentiment drift over time for a product
- [ ] Competitive comparison — analyse two products side by side
- [ ] Export reports as PDF
- [ ] REST API endpoint (`/analyze`) for programmatic access
- [ ] Docker container for one-command deployment
- [ ] User authentication + persistent cloud history

---

## Models Used

| Model | Task | Source |
|---|---|---|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment analysis | HuggingFace |
| `facebook/bart-large-mnli` | Zero-shot classification | HuggingFace |

---

## License

MIT License — free for academic and commercial use.
