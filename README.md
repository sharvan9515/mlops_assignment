# MLOps Assignment 2 — Fine-Tuning DistilBERT for Book Genre Classification

Fine-tuning a `distilbert-base-cased` model on Goodreads reviews to classify book genres across 8 categories, with a complete MLOps pipeline covering GPU-accelerated training, experiment tracking via Weights & Biases, and model deployment to Hugging Face Hub.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Model Architecture](#model-architecture)
- [Baseline Comparison](#baseline-comparison)
- [Training Configuration](#training-configuration)
- [MLOps Components](#mlops-components)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Reproduction](#setup--reproduction)
- [Links](#links)

---

## Problem Statement

Given a Goodreads book review (free text), predict the **genre** of the book from 8 categories:

| Genre |
|-------|
| Poetry |
| Children |
| Comics & Graphic |
| Fantasy & Paranormal |
| History & Biography |
| Mystery, Thriller & Crime |
| Romance |
| Young Adult |

---

## Dataset

- **Source**: [UCSD Goodreads Dataset](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/) — one `.json.gz` file per genre
- **Loading**: Top 10,000 reviews per genre fetched via streaming HTTP; 2,000 randomly sampled per genre
- **Split**: 800 train / 200 test per genre
  - Total training samples: **6,400**
  - Total test samples: **1,600**
- **Input**: `review_text` field (truncated to 512 tokens)

---

## Pipeline Overview

```
Raw Goodreads Reviews (8 genres, UCSD)
        │
        ▼
  Data Loading & Sampling
  (streaming .json.gz, 2000/genre)
        │
        ▼
  Baseline Model
  TF-IDF + Logistic Regression
        │
        ▼
  DistilBERT Tokenization
  (distilbert-base-cased, max_length=512)
        │
        ▼
  Fine-Tuning on Kaggle GPU (T4 x2)
  (HuggingFace Trainer API, 3 epochs)
        │
        ├──► Experiment Tracking (Weights & Biases)
        │
        ▼
  Evaluation & Classification Report
        │
        ▼
  Model Deployment → Hugging Face Hub
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Base model | `distilbert-base-cased` |
| Task head | `DistilBertForSequenceClassification` |
| Number of labels | 8 |
| Max token length | 512 |
| Framework | HuggingFace Transformers + PyTorch |

---

## Baseline Comparison

A TF-IDF + Logistic Regression baseline was trained on the same train/test split as a reference point.

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression (baseline) | 0.55 | 0.55 |
| DistilBERT (fine-tuned) | **0.57** | **0.57** |

DistilBERT outperforms the baseline despite the complexity of the 8-class task and noisy review text.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Train batch size | 10 |
| Eval batch size | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Optimizer | AdamW (default via Trainer) |
| Evaluation strategy | Every 100 steps |

---

## MLOps Components

### Experiment Tracking — Weights & Biases

- All training runs logged to W&B via `report_to="wandb"` in `TrainingArguments`
- Metrics tracked: training loss, eval loss, accuracy per step/epoch
- Artifacts: evaluation report logged as a W&B artifact
- W&B Dashboard: [https://wandb.ai/sharvanvittala9515-self/mlops-assignment2/overview](https://wandb.ai/sharvanvittala9515-self/mlops-assignment2/overview)

### Training Platform — Kaggle Notebooks

- Hardware: **T4 x2 GPU** (free tier)
- Internet enabled for package installation and HuggingFace Hub push
- Secrets (`WANDB_API_KEY`, `HF_TOKEN`) stored via Kaggle Secrets — no hardcoded credentials
- Kaggle Notebook: [https://www.kaggle.com/code/g25ait2099/mlops-assignment](https://www.kaggle.com/code/g25ait2099/mlops-assignment)

### Model Deployment — Hugging Face Hub

- Fine-tuned model pushed to the HuggingFace Hub for public access and inference
- Model: [https://huggingface.co/sharvanvittala007/distilbert-goodreads-genres](https://huggingface.co/sharvanvittala007/distilbert-goodreads-genres)

---

## Results

### Classification Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.57 |
| Weighted F1 | 0.57 |

### Confusion Analysis

The pipeline generates two confusion heatmaps:
1. Full confusion matrix across all 8 genres
2. Misclassification-only heatmap (off-diagonal counts)

These reveal which genre pairs are most commonly confused (e.g., fantasy/paranormal vs. young adult).

---

## Project Structure

```
.
├── fine_tuning_classification.py   # Original baseline script: data loading, TF-IDF baseline, DistilBERT fine-tuning, evaluation (no W&B, no HF push)
├── mlops_assignment.py             # MLOps version downloaded from Kaggle notebook — adds W&B experiment tracking, Kaggle Secrets, F1 metric, and Hugging Face Hub deployment
├── mlops-assignment.ipynb          # Kaggle notebook — source of mlops_assignment.py; run on T4 x2 GPU
├── MLOps_Assignment2_Updated.docx  # Assignment report
├── requirements.txt                # Python dependencies
└── README.md
```

> `mlops_assignment.py` was exported directly from the Kaggle notebook (`File → Download → Download .py`) and is the script used for the actual MLOps training run.

---

## Setup & Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/sharvan9515/mlops_assignment.git
cd mlops_assignment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

> On Kaggle, use **Add-ons → Secrets** to store these keys instead of environment variables.

### 4. Run the pipeline

```bash
python fine_tuning_classification.py
```

> GPU is required for fine-tuning. The script uses `cuda` by default. For CPU-only runs, change `device_name = 'cpu'` in the parameters section.

---

## Links

| Resource | URL |
|----------|-----|
| Hugging Face Model | [distilbert-goodreads-genres](https://huggingface.co/sharvanvittala007/distilbert-goodreads-genres) |
| W&B Dashboard | [mlops-assignment2](https://wandb.ai/sharvanvittala9515-self/mlops-assignment2/overview) |
| Kaggle Notebook | [mlops-assignment](https://www.kaggle.com/code/g25ait2099/mlops-assignment) |
| GitHub Repository | [mlops_assignment](https://github.com/sharvan9515/mlops_assignment) |
