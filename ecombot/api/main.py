"""
EcomBot — FastAPI backend
Endpoints:
  GET  /           → web UI
  POST /chat       → generate reply using LoRA model + catalog/order context
  GET  /products   → list products (for UI)
  GET  /orders     → list orders (for UI)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecombot.search.catalog import (
    build_context, filter_products_by_intent, format_product_list, PRODUCTS, ORDERS
)

app = FastAPI(title="EcomBot API")

# ── Model config ──────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "adapters/ecom-lora"

SYSTEM = (
    "You are EcomBot, a professional and empathetic customer service agent for an online store. "
    "You handle orders, returns, refunds, and product queries. "
    "Always acknowledge the customer's concern first, then provide clear and actionable help. "
    "Keep responses concise, warm, and solution-focused. "
    "CRITICAL RULES — follow these strictly:\n"
    "1. ONLY mention products that appear in the RELEVANT PRODUCTS section of the context. "
    "   Never invent, assume, or add products that are not listed there.\n"
    "2. ONLY use prices, specs, and order details from the context. Never guess or make up numbers.\n"
    "3. If the context has 3 products, list exactly those 3 — do not invent more.\n"
    "4. If no products match, say so honestly and ask the customer to clarify."
)

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading EcomBot model...")
_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
_tokenizer.pad_token = _tokenizer.eos_token
_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map="cpu")
_model = PeftModel.from_pretrained(_base, ADAPTER_PATH)
_model.eval()
print("EcomBot ready.")


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{role: 'user'|'assistant', content: str}]


# ── Helper: generate one LLM response ────────────────────────────────────────
def llm_generate(messages: list[dict], max_tokens: int = 500) -> str:
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=_tokenizer.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new, skip_special_tokens=True).strip()


# ── Step 1: ask model to extract structured intent from user query ────────────
def extract_intent(user_message: str) -> dict:
    """
    Ask the local model to parse the query into JSON with 5 fields.
    Returns a dict with: intent, category, max_price, brand, keywords
    """
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                'Extract shopping intent from user queries. Return ONLY a JSON object with 5 fields.\n\n'
                'Example:\n'
                'User: "Show me laptops under ₹1.5 lakh"\n'
                'JSON: {"intent": "list", "category": "laptops", "max_price": 150000, "brand": null, "keywords": null}\n\n'
                'User: "Do you have Sony headphones?"\n'
                'JSON: {"intent": "search", "category": "headphones", "max_price": null, "brand": "sony", "keywords": null}\n\n'
                'User: "My order ORD-1001 is delayed"\n'
                'JSON: {"intent": "order", "category": null, "max_price": null, "brand": null, "keywords": "ORD-1001"}\n\n'
                'Rules: intent is list/search/order/return/general. Use null when field is not mentioned. '
                'Return ONLY the JSON, no explanation.'
            )
        },
        {"role": "user", "content": user_message}
    ]

    raw = llm_generate(extraction_prompt, max_tokens=500)

    print(raw)

    # Extract JSON from response
    import re, json
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # Fallback if parsing fails
    return {"intent": "general", "category": None, "max_price": None, "brand": None, "keywords": user_message}



# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    # Step 1: parse intent from user message
    intent_data = extract_intent(req.message)
    print(f"Intent: {intent_data}")   # visible in server terminal for debugging

    # Step 2: if it's a list/search query → filter products, bypass LLM for data
    if intent_data.get("intent") in ("list", "search") and (
        intent_data.get("category") or intent_data.get("brand") or intent_data.get("keywords")
    ):
        products = filter_products_by_intent(intent_data)

        if products:
            product_list = format_product_list(products)
            # Ask LLM only for the 1-sentence intro
            intro_messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": f"I found {len(products)} products for you:\n\n"},
            ]
            intro = llm_generate(intro_messages, max_tokens=40)
            # Clean up: extract just the intro sentence
            intro_line = intro.split("\n")[0].strip() if intro else f"Here are {len(products)} products:"
            reply = f"{intro_line}\n\n{product_list}"
        else:
            reply = "I couldn't find any products matching your query. Could you try different filters?"

        return {"reply": reply, "products": len(products if products else []), "orders": 0, "context": str(intent_data)}

    # Step 3: conversational query → inject context + full LLM generation
    context, items_found = build_context(req.message)
    system = SYSTEM
    if context:
        system += f"\n\n{context}"

    messages = [{"role": "system", "content": system}]
    messages.extend(req.history[-6:])
    messages.append({"role": "user", "content": req.message})

    reply = llm_generate(messages, max_tokens=200)

    return {
        "reply":    reply,
        "products": items_found,
        "orders":   items_found,
        "context":  context,
    }


# ── Data endpoints ────────────────────────────────────────────────────────────
@app.get("/products")
async def get_products():
    return PRODUCTS[:20]

@app.get("/orders")
async def get_orders():
    return ORDERS[:10]


# ── Web UI ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>EcomBot — Customer Service</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

    header { padding: 1rem 1.5rem; background: #141414; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 0.75rem; }
    .logo { width: 32px; height: 32px; background: linear-gradient(135deg, #4a9eff, #a855f7); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem; }
    header h1 { font-size: 1rem; font-weight: 600; color: #fff; }
    header p  { font-size: 0.75rem; color: #555; }
    .status-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-left: auto; }

    .chat-area { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }

    .msg { display: flex; gap: 0.75rem; max-width: 85%; }
    .msg.user  { align-self: flex-end; flex-direction: row-reverse; }
    .msg.bot   { align-self: flex-start; }

    .avatar { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 600; }
    .msg.user  .avatar { background: #1e3a5a; color: #4a9eff; }
    .msg.bot   .avatar { background: linear-gradient(135deg, #4a9eff, #a855f7); color: white; font-size: 0.7rem; }

    .bubble { padding: 0.75rem 1rem; border-radius: 12px; font-size: 0.9rem; line-height: 1.55; max-width: 100%; white-space: pre-wrap; }
    .msg.user .bubble { background: #1c2a3a; color: #c8dff8; border-radius: 12px 12px 2px 12px; }
    .msg.bot  .bubble { background: #1a1a1a; color: #e0e0e0; border-radius: 12px 12px 12px 2px; border: 1px solid #2a2a2a; }
    .msg.bot  .bubble .ctx { font-size: 0.72rem; color: #444; margin-top: 0.5rem; border-top: 1px solid #2a2a2a; padding-top: 0.4rem; }
    .thinking { color: #444; font-style: italic; }

    .input-area { padding: 1rem 1.5rem; background: #141414; border-top: 1px solid #222; display: flex; gap: 0.75rem; }
    .input-area input { flex: 1; padding: 0.75rem 1rem; background: #1a1a1a; border: 1px solid #333; border-radius: 10px; color: #e0e0e0; font-size: 0.95rem; outline: none; }
    .input-area input:focus { border-color: #4a9eff; }
    .input-area button { padding: 0.75rem 1.25rem; background: #4a9eff; border: none; border-radius: 10px; color: white; font-weight: 600; cursor: pointer; font-size: 0.9rem; }
    .input-area button:hover { background: #357acc; }

    .welcome { text-align: center; padding: 3rem 1rem; color: #444; }
    .welcome h2 { color: #666; font-size: 1rem; margin-bottom: 0.5rem; }
    .suggestions { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; margin-top: 1.5rem; }
    .chip { padding: 0.4rem 0.9rem; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; font-size: 0.8rem; color: #888; cursor: pointer; }
    .chip:hover { border-color: #4a9eff; color: #4a9eff; }
  </style>
</head>
<body>
  <header>
    <div class="logo">🛍</div>
    <div>
      <h1>EcomBot</h1>
      <p>Customer Service · Returns · Orders · Products</p>
    </div>
    <div class="status-dot" title="Online"></div>
  </header>

  <div class="chat-area" id="chat">
    <div class="welcome">
      <h2>Hi! I'm EcomBot 👋</h2>
      <p>Ask me about your orders, returns, or products.</p>
      <div class="suggestions">
        <span class="chip" onclick="sendSuggestion(this)">Where is my order?</span>
        <span class="chip" onclick="sendSuggestion(this)">I want to return an item</span>
        <span class="chip" onclick="sendSuggestion(this)">Best laptop for coding under ₹1.5L?</span>
        <span class="chip" onclick="sendSuggestion(this)">My package arrived damaged</span>
        <span class="chip" onclick="sendSuggestion(this)">Recommend noise-cancelling headphones</span>
      </div>
    </div>
  </div>

  <div class="input-area">
    <input id="msg" type="text" placeholder="Type your message..." autocomplete="off"/>
    <button onclick="send()">Send</button>
  </div>

<script>
let history = [];
const chat = document.getElementById('chat');
const msgInput = document.getElementById('msg');

msgInput.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

function sendSuggestion(el) {
  msgInput.value = el.textContent;
  send();
}

function addBubble(role, text, meta) {
  // Remove welcome screen on first message
  const welcome = chat.querySelector('.welcome');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const avatar = role === 'user' ? 'U' : '🛍';
  let bubble = `<div class="bubble">${text.replace(/\\n/g, '<br>')}`;
  if (meta) bubble += `<div class="ctx">${meta}</div>`;
  bubble += `</div>`;
  div.innerHTML = `<div class="avatar">${avatar}</div>${bubble}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function send() {
  const text = msgInput.value.trim();
  if (!text) return;
  msgInput.value = '';

  addBubble('user', text);

  const thinking = addBubble('bot', '<span class="thinking">Thinking...</span>', null);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: text, history })
    });
    const data = await res.json();

    thinking.remove();
    const meta = data.context
      ? `📦 ${data.products} item(s) referenced from catalog`
      : null;
    addBubble('bot', data.reply, meta);

    history.push({ role: 'user',      content: text });
    history.push({ role: 'assistant', content: data.reply });

  } catch (e) {
    thinking.remove();
    addBubble('bot', 'Sorry, something went wrong. Please try again.', null);
  }
}
</script>
</body>
</html>
"""
