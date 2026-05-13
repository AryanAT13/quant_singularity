# AI-SLM Signal Pod 

**Track:** AI-SLM · **Asset:** NIFTY 50 Options 

---

## Overview

The system is a production-style signal pod that ingests a NIFTY options market state snapshot and emits a structured trading signal through a deterministic orchestration layer.

The pod was built under the constraint that **the orchestrator must always win over the model.** Every output — whether a valid directional signal or a forced NEUTRAL — is the result of the deterministic layer, never raw model output. The model earns its place only after every rule-based check has been satisfied.

---

## Kaggle Fine-Tuning Notebook

All model training was executed on Kaggle free-tier (T4 GPU, 30 hrs/week). No external GPU compute was used.

**Notebook URL:** `https://www.kaggle.com/code/aryanamittiwari/damnnnn`

The notebook is also committed to this repository at `notebook/quant_finetune.ipynb` for offline reference. MLflow run timestamps in `mlruns/` are verifiable against Kaggle execution timestamps.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── report.pdf                   
│
├── data/
│   ├── raw/
│   │   ├── market_states.parquet         # 60-day NIFTY options data
│   │   ├── finetune_instructions.jsonl   # Raw instruction file
│   │   ├── rag_corpus.jsonl              # RAG episode corpus
│   │   └── retrieve.py                   # Retrieval function
│   └── processed/
│       └── finetune_clean_cot.jsonl         # Audited/sanitized dataset
│
├── src/
│   ├── data_prep.py                  # Data audit + sanitization pipeline
│   ├── orchestrator.py               # Pydantic signal schema + deterministic rules
│   ├── pipeline.py                   # Walk-forward evaluation engine (Days 31–60)
│   └── evaluation.py                 # Eval suite metrics engine
│
├── notebook/
│   └── quant_finetune.ipynb          # Fine-tuning notebook (mirrors Kaggle)
│
├── quant_slm_adapter/                # LoRA adapter weights (Rank 8, Alpha 16)
│
└── mlruns/                           # MLflow tracking (immutable)

```
---
## System Architecture
 
```
Market State Snapshot
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                     │
│                                                     │
│  Rule 1: ADX < 20  ──► NEUTRAL (ADX_SUPPRESSION)    │
│          │                                          │
│          ▼                                          │
│  Rule 2: Call SLM Pod                               │
│          │                                          │
│          ▼                                          │
│  Rule 3: Parse failure ──► NEUTRAL (PARSE_ERROR)    │
│          │                                          │
│          ▼                                          │
│  Rule 4: Conviction < 0.40 ──► NEUTRAL (LOW_CONV)   │
│          │                                          │
│          ▼                                          │
│     Valid Signal (CE / PE / NEUTRAL)                │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Downstream Pipeline
```
Every rule logs a reason code and the triggering value. The downstream pipeline never receives raw model output under any condition.

---

## Signal Schema & Conviction Design

All outputs conform to the following fixed schema:

```python
class SignalSchema(BaseModel):
    direction:  Literal["CE", "PE", "NEUTRAL"]
    conviction: float          
    horizon:    Literal["intraday", "next_session"]
    signal_id:  str            
    timestamp:  str
```

**On the conviction field:** Relying on LLM softmax token probabilities for financial risk is fundamentally unsafe. Instead, Conviction is generated deterministically. During data prep, a mathematical Confluence Score (evaluating ADX, VIX, and PCR) was injected into a Chain-of-Thought analysis block inside the target JSON. The model was fine-tuned to explicitly evaluate the market state in this text block before outputting the final float conviction.

---

## Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base model | TinyLlama-1.1B-Chat | Fits Kaggle T4 memory; highly efficient for CPU inference. |
| Fine-tuning method | LoRA only | Brief constraint |
| LoRA rank | 8 | Balances expressivity vs. overfitting on noisy financial datasets. |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` | Attention layers; sufficient for instruction following |
| Inference | MPS/CPU, float16/4-bit quantization | Brief constraint |

---

## Evaluation Design & RAG Ablation

Walk-forward evaluation was conducted exclusively on Days 31–60 in strict 5-day rolling blocks.

A two-condition ablation was run across the full walk-forward evaluation set:

- **Control:** No retrieval context
- **Treatment:** 3 retrieved episodes via `retrieve.py` prepended to the prompt

### Ablation Result

The Control run accurately parsed schemas but occasionally hallucinated contextually-adjacent tokens (e.g., `CELEBRATE` instead of `CE`), which the Orchestrator safely blocked.

The RAG Treatment caused catastrophic **"Context Shock"** in the 1.1B SLM. Inundated with historical paragraphs, the model lost instruction-following adherence and experienced a nearly 100% schema failure rate.

The defensive Orchestrator caught all errors, successfully maintaining a `0.0 Conviction NEUTRAL` safety state.

---

## Setup & Reproduction

### Requirements

```bash
Python 3.10+
```

### Install

```bash
pip install -r requirements.txt
```

### Run Full Walk-Forward Evaluation (with MLflow tracking)

```bash
python3 -m src.pipeline
```

This runs the full Day 31–60 walk-forward evaluation through the orchestrator and writes per-window results to `results/walk_forward_results.json`.
