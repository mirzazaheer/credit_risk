# E-commerce MSME Credit Risk Screener
**C2_AIB0326 — Team 1**

An AI-powered credit risk screening tool for Indian e-commerce MSMEs using alternative data signals — GST compliance, marketplace performance, and payment behaviour — instead of traditional credit scores.

---

## What it does

1. Takes MSME business data as input (GST filings, seller ratings, payment history, etc.)
2. Computes a **Credit Risk Score (0–100)** using a trained Random Forest model
3. Classifies into **Low / Medium / High Risk** tier
4. Shows key **positive and negative factors** driving the score
5. Generates a **human-readable credit assessment report** via Llama 3 (local LLM)

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Internet connection for first-time setup (Ollama ~200 MB + Llama 3 model ~4 GB)

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run setup (one-time)
```bash
python setup.py
```
This automatically:
- Generates 1,200 synthetic MSME training records
- Trains the Random Forest credit risk model (~30 seconds)
- Downloads Ollama to `bin/` (LLM runtime, ~200 MB)
- Pulls the Llama 3 model (~4 GB) for AI narrative reports

To skip the LLM step (app still works, uses rule-based reports):
```bash
python setup.py --skip-ollama
```

### Step 3 — Launch the app

**macOS / Linux:**
```bash
bin/ollama serve          # Terminal 1 — keep open
streamlit run app.py      # Terminal 2
```

**Windows:**
```bat
bin\ollama.exe serve
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> If Ollama is already installed system-wide, use `ollama serve` instead of `bin/ollama serve`.

> Without Ollama running the app still works — it falls back to a structured rule-based credit report automatically.

---

## Project Structure

```
CREDIT_RISK/
├── data/
│   └── generate_synthetic.py   # Synthetic MSME dataset generator
├── features/
│   └── engineer.py             # GST / Payment / Marketplace score computation
├── model/
│   ├── train.py                # Random Forest training
│   ├── predict.py              # Inference pipeline
│   └── explainer.py            # Feature importance explainability
├── llm/
│   └── report_generator.py     # Llama 3 (Ollama) narrative report generator
├── app.py                      # Streamlit web UI
├── setup.py                    # One-command bootstrap (run once after cloning)
└── requirements.txt
```

---

## Data Sources (this prototype)

Since real-time GST/marketplace APIs require institutional access, this prototype uses **synthetic data** that mirrors realistic MSME profiles across three risk cohorts. This is consistent with the project charter (Section 6 — Risks/Limitations).

In production, the same pipeline would ingest data from:
- GST Portal (`services.gst.gov.in`)
- Udyam MSME Registration (`udyamregistration.gov.in`)
- Marketplace seller dashboards (Amazon, Flipkart, Myntra)
- Payment/invoice records

---

## System Architecture

```
MSME Input Data
      |
      v
Feature Engineering (GST + Payment + Marketplace scores)
      |
      v
Random Forest Classifier  ──►  Risk Score (0–100) + Tier
      |
      v
Feature Importance Explainer  ──►  Positive / Negative Factors
      |
      v
Llama 3 (Ollama)  ──►  Human-readable Credit Assessment Report
```

---

## Requirements

- Python 3.10–3.13 recommended (3.14 works with minor package warnings)
- No paid API keys needed
- Ollama optional (for AI narrative reports)
