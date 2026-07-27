# EcomBot 🛍

An AI-powered e-commerce customer service bot — fine-tuned on real customer service patterns using LoRA, with a hybrid generation pipeline that prevents hallucination.

---

## Demo

<!-- Add demo video/GIF here -->

---

## How It Works

```
User query
    ↓
Local LLM (Qwen 0.5B + LoRA) extracts intent as JSON
{"intent": "list", "category": "smartphones", "brand": null, ...}
    ↓
Python filters products.json / orders.json — no LLM, no hallucination
    ↓
Clean, accurate response
```

**Three query paths:**
- **Product listing/search** → intent extraction → Python filter → formatted list
- **Order tracking** → order ID regex → `orders.json` lookup → order details
- **Customer service** (returns, complaints) → LLM generates empathetic response with order context

---

## Stack

| Layer | Tech |
|-------|------|
| Base model | Qwen 2.5 0.5B Instruct |
| Fine-tuning | LoRA (r=16, α=32) via PEFT + TRL SFTTrainer |
| Training data | 35 seeds × 10 Kimi variations = 385 examples |
| Backend | FastAPI |
| Data | `products.json` (20 products) + `orders.json` (mock orders) |
| Inference | Local CPU — no cloud API at serving time |

---

## Fine-Tuning

The adapter was trained on 3 domains:
- **Order & Shipping** (12 seeds) — delays, damage, cancellations, tracking
- **Returns & Refunds** (12 seeds) — return flow, exchange, store credit
- **Product & Shopping** (11 seeds) — recommendations, comparisons, availability

Seeds defined response *patterns* — tone, empathy, structure. Product facts come from the catalog at runtime, not from the model's weights.

---

## Setup

```bash
git clone <repo>
cd ecom-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add KIMI_API_KEY (only needed for dataset generation, not for serving)
```

**Run:**
```bash
PYTHONPATH=. uvicorn ecombot.api.main:app --reload --port 8001
```

Open **http://127.0.0.1:8001**

First startup takes ~30 seconds (loads Qwen + LoRA adapter into memory).

---

## Project Structure

```
ecom-bot/
├── ecombot/
│   ├── api/main.py          — FastAPI + web UI + intent extraction
│   └── search/catalog.py    — filter_products_by_intent, order lookup
├── data/
│   ├── products.json        — product catalog (20 items)
│   └── orders.json          — mock orders
├── scripts/
│   ├── generate_dataset.py  — generates training data via Kimi API
│   └── train_lora.py        — fine-tunes Qwen 0.5B with LoRA
├── adapters/ecom-lora/      — trained LoRA weights (~4MB)
└── clarify.md               — technical notes and Q&A
```