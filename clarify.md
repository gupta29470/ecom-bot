# Adapter Files — What Each One Is

> These files live in `adapters/sid-lora/` after running `train_lora.py`.

---

## 1. checkpoint-50 and checkpoint-63 — which is the "finish line"?

**checkpoint-50** was saved automatically at step 50 because you set `save_steps=50` in SFTConfig.
**checkpoint-63** was saved at the final step (63 total steps = 3 epochs × 21 steps per epoch).

But neither checkpoint is the model you actually use for inference.

The **root `adapters/sid-lora/` directory** (not a checkpoint subfolder) is what matters. That was explicitly written at the end of training by:
```python
model.save_pretrained(OUTPUT_DIR)   # → adapters/sid-lora/adapter_model.safetensors
```

```
adapters/sid-lora/
├── adapter_model.safetensors   ← USE THIS for inference (final trained weights)
├── adapter_config.json
├── checkpoint-50/              ← mid-training snapshot (recovery only)
│   └── adapter_model.safetensors  (weights at step 50, not fully trained)
└── checkpoint-63/              ← same as root but with full optimizer state
    ├── adapter_model.safetensors
    ├── optimizer.pt            ← Adam m and v states (needed to RESUME training)
    └── scheduler.pt            ← LR scheduler state
```

**Rule:** Load from `adapters/sid-lora/` (root), not from a checkpoint subfolder, unless you are resuming training.

---

## 2. adapter_config.json — is this the LoraConfig params we set?

**Yes, exactly.** It's a serialised copy of the `LoraConfig(...)` you passed to `get_peft_model()`.

```json
{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

**Why it's needed:** When you call `PeftModel.from_pretrained(base, 'adapters/sid-lora')`, PEFT reads this file first to know:
- How many LoRA layers to inject (r=16)
- Which layers to target (q_proj, k_proj, v_proj, o_proj)
- How to scale the output (alpha/r = 2.0)

Without this file, PEFT cannot reconstruct the adapter structure to load the weights into.

---

## 3. adapter_model.safetensors — what is this?

**This IS Sid's personality.** It contains the actual trained A and B matrices.

```
192 tensors total:
  4 target modules × 24 transformer layers × 2 matrices (A, B) = 192

Each tensor:
  lora_A: shape (16, 896)   → 14,336 values
  lora_B: shape (896, 16)   → 14,336 values

Total: 2,162,688 float16 values ≈ 4MB on disk
```

**Why safetensors and not .pt / pickle?**
- `.pt` (pickle) can execute arbitrary Python code when loaded → security risk
- safetensors: zero-code, zero-copy memory mapping, header-first (can read metadata without loading all weights)
- Same format used by HuggingFace Hub for all models

This file is the ONLY thing you need to share Sid's personality. Anyone who has:
1. `Qwen/Qwen2.5-0.5B-Instruct` (public, downloadable)
2. `adapter_model.safetensors` + `adapter_config.json`

...can load Sid locally with `PeftModel.from_pretrained()`.

---

## 4. chat_template.jinja — what is this?

**A Jinja2 template that defines the exact format for system/user/assistant conversations.**

Qwen2.5 expects messages in a specific format. When you call:
```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
...this Jinja template is rendered to produce the correctly-formatted string.

It handles:
- Which special tokens wrap each role (`<|im_start|>user`, `<|im_end|>`, etc.)
- Whether to add a BOS token at the start
- Whether to append the assistant prefix at the end (for generation)

**Why it matters:** If you format the prompt differently than the template expects, the model gives degraded output because it was pretrained on this exact format. Your training data was formatted using this same template — so loading and running inference with it ensures consistency.

You rarely edit this file directly — it's auto-copied from the base model's tokenizer.

---

## 5. tokenizer_config.json — is this the tokenizer settings from our code?

**Yes — plus the base model's defaults.** It stores:

```json
{
  "model_max_length": 131072,
  "pad_token": "<|endoftext|>",     ← we set this: tokenizer.pad_token = tokenizer.eos_token
  "eos_token": "<|endoftext|>",
  "bos_token": null,
  "tokenizer_class": "Qwen2Tokenizer",
  "chat_template": "..."            ← reference to the jinja template
}
```

The settings we set in code (`tokenizer.pad_token = tokenizer.eos_token`) are persisted here so they're automatically restored next time you do:
```python
tokenizer = AutoTokenizer.from_pretrained('adapters/sid-lora')
# pad_token is already set correctly — no need to set it again
```

---

## 6. tokenizer.json — is this the whole history of the tokenizer?

**No — it's the complete tokenizer algorithm in serialized form.** Not history.

It contains three things:

### a) Vocabulary — all tokens and their IDs
```json
"vocab": {
  "!": 0,
  "\"": 1,
  ...
  "Ġhello": 12345,
  ...
}
```
Qwen2.5 has ~150,000 tokens. Each token is a string → integer mapping.

### b) BPE merge rules — how subword pieces are combined
```json
"merges": [
  ["Ġ", "t"],      → "Ġt"
  ["Ġt", "he"],    → "Ġthe"
  ...
]
```
"transformer" might tokenize to `["transform", "er"]` — the merge rules define this splitting.

### c) Pre-tokenization patterns — how text is split before BPE
Regex patterns that split on whitespace, punctuation, etc. before BPE applies.

**Why it's copied to the adapter folder:** When you save an adapter, `tokenizer.save_pretrained()` copies the full tokenizer so the adapter is self-contained. Anyone loading `adapters/sid-lora` gets the exact tokenizer Sid was trained with, without needing a separate download.

---

## Quick Reference

| File | What it is | Needed for inference? |
|------|-----------|----------------------|
| `adapter_model.safetensors` | The trained A and B weights (Sid's personality) | ✅ Yes |
| `adapter_config.json` | LoraConfig params (r, alpha, target_modules) | ✅ Yes |
| `tokenizer.json` | Full BPE vocabulary + merge rules | ✅ Yes |
| `tokenizer_config.json` | Tokenizer settings (pad token, max length) | ✅ Yes |
| `chat_template.jinja` | Conversation format template | ✅ Yes |
| `checkpoint-50/` | Mid-training snapshot (step 50) | ❌ No (recovery only) |
| `checkpoint-63/` | Final step + optimizer state | ❌ No (resume training only) |

---

# Inference Code — Explained Line by Line

This covers the code in `scripts/test_adapter.py` and `brocode/voice/loop.py`.

---

## 1. Why `tokenize=False` in apply_chat_template, then tokenize separately?

```python
# Step 1: format into a string
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# prompt is now a plain Python string:
# "<|im_start|>system\nYou are Sid...<|im_end|>\n<|im_start|>user\nHow was my day?<|im_end|>\n<|im_start|>assistant\n"

# Step 2: tokenize the string into tensors
inputs = tokenizer(prompt, return_tensors="pt")
# inputs = {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}
```

**If you used `tokenize=True` in step 1:**
```python
ids = tokenizer.apply_chat_template(messages, tokenize=True)
# ids is a plain Python LIST of ints: [151644, 8948, 198, ...]
# No attention_mask. No batch dimension. Cannot pass directly to model.generate().
```

**They are NOT the same.** The two-step approach gives you:
- A formatted string you can inspect/debug (`print(prompt)`)
- A proper `BatchEncoding` dict with both `input_ids` AND `attention_mask` as PyTorch tensors
- The correct shape `(1, seq_len)` with a batch dimension that `model.generate()` expects

---

## 2. What is `add_generation_prompt=True`?

It appends the **assistant turn opening** to the end of the formatted prompt.

```python
# add_generation_prompt=False (default):
# "...<|im_end|>\n"
# Model sees end of conversation. Might stop or echo.

# add_generation_prompt=True:
# "...<|im_end|>\n<|im_start|>assistant\n"
#                  ^^^^^^^^^^^^^^^^^^^^^^^^
#                  The model continues from HERE
```

**Without it:** The prompt ends at the close of the user turn. The model doesn't know it's supposed to reply.

**With it:** The prompt ends mid-sentence with the assistant prefix open. The model has no choice but to continue generating the assistant's response.

Think of it as leaving a blank line with "Assistant:" written on it — the model fills in the rest.

---

## 3. What is `return_tensors="pt"`?

Tells the tokenizer what format to return the token IDs in.

```python
tokenizer(prompt, return_tensors="pt")   # → PyTorch tensors (torch.Tensor)
tokenizer(prompt, return_tensors="tf")   # → TensorFlow tensors
tokenizer(prompt, return_tensors="np")   # → NumPy arrays
tokenizer(prompt)                        # → plain Python lists (no return_tensors)
```

We use `"pt"` because `model.generate()` expects PyTorch tensors.

```python
inputs = tokenizer(prompt, return_tensors="pt")
print(type(inputs['input_ids']))     # <class 'torch.Tensor'>
print(inputs['input_ids'].shape)     # torch.Size([1, 47])  — (batch=1, seq_len=47)
print(inputs['attention_mask'])      # tensor([[1, 1, 1, ..., 1]])  — 1=real token, 0=padding
```

---

## 4. What is `with torch.no_grad()`?

Disables gradient computation (the autograd graph) inside the block.

```python
with torch.no_grad():
    output_ids = model.generate(**inputs, ...)
```

**During training:** PyTorch records every operation to build a computation graph for `loss.backward()`.
**During inference:** You never call `.backward()`. Building the graph wastes memory and time.

| | With no_grad | Without no_grad |
|--|-------------|----------------|
| Memory | ~1× | ~2× (graph stored) |
| Speed | Faster | Slower |
| Can call .backward()? | No | Yes |

Rule: **Always use `torch.no_grad()` for inference.** `torch.inference_mode()` is even stronger (slightly faster) and is preferred for pure inference.

---

## 5. What does `model.generate()` return, and how do we extract the result?

`model.generate()` returns the **complete sequence** — input tokens + newly generated tokens joined together.

```python
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
# output_ids shape: (1, input_length + new_tokens_length)
# e.g.            : (1, 47 + 23) = (1, 70)
#                        ↑ prompt   ↑ Sid's reply
```

**Extracting only Sid's reply:**
```python
# output_ids[0]  → shape (70,)  — remove batch dimension (we only have 1 sequence)
# inputs['input_ids'].shape[1] → 47  — how long the input (prompt) was

new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
#                          ↑ skip the first 47 tokens (the prompt)
#            leaving only the 23 new tokens Sid generated

reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
# skip_special_tokens=True removes <|im_end|> and similar from the output string
```

**Visualised:**
```
output_ids[0]:
[ 151644, 8948, 198, ...(prompt tokens x47)..., 35517, 198, 13, 0 ]
 └───────────────────────────────────────────┘ └──────────────────┘
         input prompt (47 tokens)               Sid's reply (23 tokens)

new_tokens = output_ids[0][47:]   → [35517, 198, 13, 0]
tokenizer.decode([35517, 198, 13, 0]) → "One commit? You're still writing code."
```

---

## 6. What is `TaskType.CAUSAL_LM`?

**CAUSAL_LM = Causal Language Model.** It tells PEFT what kind of task the base model does, so it configures the LoRA adapter correctly.

### What "causal" means

A causal language model predicts the **next token** using only the tokens that came **before** it — never tokens that come after.

```
Input:   "I only wrote one"
Predict: "commit"   ← can only look LEFT (at previous tokens)

Input:   "I only wrote one commit"
Predict: "today"    ← still only looks LEFT
```

**"Causal" = each position can only attend to itself and earlier positions.**

This is the architecture of every GPT-style model: GPT-2, GPT-4, Llama, Mistral, Qwen. They generate text one token at a time, always left-to-right, never peeking at future tokens.

### Contrast with BERT-style (non-causal / masked LM)

BERT uses **bidirectional attention** — it can look at tokens on both sides:
```
Input:   "I only wrote one [MASK] today."
Predict: "commit"   ← can look LEFT and RIGHT to fill the blank
```

BERT is better for understanding (classification, NER). GPT/Qwen is better for generation.

### Why it matters for LoRA

When you set `task_type=TaskType.CAUSAL_LM`, PEFT knows:
- The model generates autoregressively (left-to-right)
- The LoRA layers should be applied to the attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) used in the causal self-attention mechanism
- The loss during training is **next-token prediction loss** (cross-entropy on the next token)

If you were fine-tuning BERT for classification, you'd use `TaskType.SEQ_CLS` instead.

### The training objective, visualised

```
Training example:
  Input:   "<|system|>You are Sid...<|user|>How was my day?<|assistant|>"
  Target:  "                                                              Git shows 0 commits. Spectacular."

Loss is computed only on the assistant tokens (the reply).
The model learns: given this conversation, predict each word of Sid's reply in order.
```

This is exactly what `SFTTrainer` does — it masks the loss on system and user tokens, only backpropagating through the assistant response tokens.

---

## 7. LoraConfig Deep Dive

### 7a. All TaskType options

```python
from peft import TaskType

TaskType.CAUSAL_LM        # GPT, Llama, Qwen, Mistral — next-token generation
TaskType.SEQ_2_SEQ_LM     # T5, BART — encoder-decoder (translation, summarisation)
TaskType.SEQ_CLS          # BERT for classification (sentiment, intent detection)
TaskType.TOKEN_CLS        # BERT for NER (named entity recognition)
TaskType.QUESTION_ANS     # Extractive QA (find answer span in a passage)
TaskType.FEATURE_EXTRACTION  # Embedding models (sentence-transformers)
```

We use `CAUSAL_LM` because Qwen2.5 is a GPT-style next-token predictor.

---

### 7b. What are target_modules?

```python
target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
```

These are the **specific weight matrices inside the transformer's attention mechanism** where LoRA injects its A and B matrices.

Inside every transformer layer there are 4 linear projections for attention:

```
Input embedding (x)
        │
        ├──→ q_proj  → Query (Q)   ← "what am I looking for?"
        ├──→ k_proj  → Key   (K)   ← "what do I contain?"
        ├──→ v_proj  → Value (V)   ← "what do I actually pass forward?"
        └──→ o_proj  → Output (O)  ← "combine all attention heads into one vector"
```

**How attention works with these:**
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
                      ↑ which tokens matter  ↑ what to read from them
Output = concat(all_heads) × o_proj
```

**Why target only these 4, not everything?**
- They are where the model decides "what to pay attention to"
- Fine-tuning attention = changing what the model focuses on → changes personality/style
- The FFN (feed-forward) layers store factual knowledge — we leave those untouched
- Targeting just q, k, v, o covers personality well with minimum params

**Wider targeting for better quality:**
```python
# Minimal (personality fine-tuning)
target_modules=['q_proj', 'v_proj']              # 2 matrices, fewer params

# Standard (good balance)
target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']  # 4 matrices ← what we used

# Full attention + FFN (highest quality, more params)
target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj']       # 7 matrices
```

---

### 7c. All LoraConfig parameters explained

```python
from peft import LoraConfig, TaskType

LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # ↑ Tells PEFT what kind of model this is.
    #   Controls how the adapter is structured and which layers are eligible.

    r=16,
    # ↑ RANK — the bottleneck dimension of A and B matrices.
    #   W is (d_out × d_in). LoRA makes it: B (d_out × r) × A (r × d_in).
    #   r=4:   minimal params, fast. Good for simple style changes.
    #   r=16:  standard. Good balance of expressiveness and efficiency.
    #   r=64:  high expressiveness. Use for complex tasks or larger datasets.
    #   Doubles params per layer when doubled.

    lora_alpha=32,
    # ↑ SCALING FACTOR — controls how strongly the adapter affects the output.
    #   Effective update = (alpha / r) × B × A
    #   With r=16, alpha=32: scale = 32/16 = 2.0
    #   Convention: alpha = 2 × r  (keeps effective scale at ~2 regardless of r)
    #   Higher alpha → adapter has more influence over base model.
    #   Lower alpha → adapter is a subtle nudge.

    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    # ↑ WHICH LAYERS to inject LoRA into (see section 7b above).

    lora_dropout=0.05,
    # ↑ DROPOUT — randomly zero out 5% of adapter activations during training.
    #   Prevents the adapter from memorising training examples (overfitting).
    #   Only active during training. Disabled at inference (model.eval()).
    #   Set to 0.0 if your dataset is large (>10K examples) — less needed.
    #   Set higher (0.1) if your dataset is very small (<50 examples).

    bias='none',
    # ↑ WHETHER to train bias terms alongside LoRA.
    #   'none':   don't train biases at all (default, most common)
    #   'all':    train all bias parameters in the model
    #   'lora_only': only train biases in the LoRA layers
    #   'none' is almost always correct for personality fine-tuning.

    # Less common but useful:
    # modules_to_save=['embed_tokens', 'lm_head'],
    # ↑ Modules to save as full weights (not LoRA) — e.g. if you added new tokens
    #   and need to save the updated embedding table.

    # init_lora_weights=True,
    # ↑ Default True: A is random Gaussian, B is zeros → ΔW=0 at start.
    #   False: both random → model starts broken, needs more training to recover.
)
```

---

### 7d. What would be different with QLoRA?

**QLoRA = quantize the base model to 4-bit, then add LoRA adapters on top.**

Everything in the adapter (A and B matrices) stays in float16. Only the frozen base weights are compressed.

```python
# QLoRA setup (needs bitsandbytes library)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',           # NormalFloat4 — best quality for weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16 even if stored in 4-bit
    bnb_4bit_use_double_quant=True,       # quantize the quantization constants too
)

base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    quantization_config=bnb_config,      # ← base weights now in 4-bit
    device_map='auto',
)

# Then add LoRA exactly as before — nothing changes in LoraConfig
model = get_peft_model(base, lora_config)
```

| | LoRA (what we did) | QLoRA |
|--|-------------------|-------|
| Base model precision | float32 (CPU) | 4-bit NF4 |
| Adapter precision | float32 | bfloat16 |
| Base model memory | ~2GB (fp32 × 0.5B) | ~350MB (4-bit × 0.5B) |
| Training quality | Slightly better | Very close to LoRA |
| Requires | Any hardware | `bitsandbytes` (NVIDIA GPU or specific setups) |
| Use case | Small models, CPU training | Large models (7B+), GPU VRAM limited |

**For our 0.5B model on CPU, LoRA was the right call.** QLoRA becomes essential when you're training a 7B+ model and can't fit the base weights in VRAM even in float16.

---

### 7e. What would be different with full fine-tuning?

**Full fine-tuning = every weight in the model is trained.** No LoRA, no freezing.

```python
# Full fine-tuning — no PEFT, no get_peft_model()
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
# All 494M params have requires_grad=True

trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, ...)
trainer.train()
# Gradients flow through ALL 494M params on every step
```

| | LoRA (what we did) | Full fine-tuning |
|--|-------------------|-----------------|
| Trainable params | 2.16M (0.44%) | 494M (100%) |
| GPU memory (0.5B) | ~2GB | ~8GB (model + grads + Adam states) |
| Training time | 59 seconds | ~10 minutes |
| Saved file size | ~4MB | ~1GB |
| Risk of forgetting base knowledge | Very low | High (catastrophic forgetting) |
| Quality ceiling | Good | Highest (can change anything) |
| Inference: load base + adapter? | Yes | No — the whole model is modified |
| Hot-swap between personalities? | ✅ Yes — `set_adapter()` | ❌ No — need 2 full 1GB models |

**The hot-swap is only possible with LoRA.** With full fine-tuning, Sid and Guru would be two completely separate 1GB models that can't share memory. With LoRA, both are 4MB adapters on top of the same 1GB base — you can load both into memory and switch instantly.

This is the architectural reason the roadmap chose LoRA over full fine-tuning for BroCode.

---

## 8. Causal vs Masked Language Modelling

Two completely different ways of training a model on text. The architecture determines what the model is good at.

---

### Causal Language Modelling (CLM) — GPT-style

**Objective:** Predict the next token given all previous tokens.

```
Input text: "Git shows 14 commits"

Training tasks:
  "Git"                        → predict "shows"
  "Git shows"                  → predict "14"
  "Git shows 14"               → predict "commits"
  "Git shows 14 commits"       → predict <end>
```

**Attention mask — each token can only see itself and tokens to its LEFT:**

```
Tokens:   Git   shows   14   commits
                                    ↑ can see all 4 tokens before it
                        ↑ can see 3 tokens before it
              ↑ can see 2 tokens before it
   ↑ can see only itself

Attention matrix (1=can attend, 0=blocked):
         Git  shows  14  commits
Git     [ 1,    0,   0,    0  ]
shows   [ 1,    1,   0,    0  ]
14      [ 1,    1,   1,    0  ]
commits [ 1,    1,   1,    1  ]
```

The upper triangle is always masked (zeroed out). This is why it's called **causal** — no token can "cheat" by looking at future tokens.

**Models:** GPT-2, GPT-3/4, Llama, Mistral, Qwen, Falcon, Claude
**Use case:** Text generation, conversation, completion, coding

---

### Masked Language Modelling (MLM) — BERT-style

**Objective:** Predict a randomly hidden (masked) token using ALL surrounding tokens — both left and right.

```
Original: "Git shows 14 commits today"
Masked:   "Git shows [MASK] commits today"

Task: predict what [MASK] is → "14"
The model can look at "commits today" (future) AND "Git shows" (past)
```

**Attention mask — every token can see every other token (bidirectional):**

```
Attention matrix (1=can attend):
         Git  shows  14  commits  today
Git     [ 1,    1,   1,    1,      1  ]
shows   [ 1,    1,   1,    1,      1  ]
14      [ 1,    1,   1,    1,      1  ]
commits [ 1,    1,   1,    1,      1  ]
today   [ 1,    1,   1,    1,      1  ]
```

All ones — full bidirectional attention. Every token sees the whole sentence.

**Models:** BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa
**Use case:** Classification, NER, question answering, semantic similarity, embeddings

---

### Side-by-side comparison

| | Causal (CLM) | Masked (MLM) |
|--|--------------|--------------|
| What it predicts | Next token from left context | Missing token from full context |
| Attention direction | Left only (unidirectional) | Both directions (bidirectional) |
| Good at | **Generating** new text | **Understanding** existing text |
| Examples | GPT-4, Llama, Qwen | BERT, RoBERTa |
| Can generate text? | ✅ Yes — one token at a time | ❌ Not directly |
| Can classify text? | ✅ With a head on top | ✅ Better at it natively |
| Training signal | Every token predicts next | Only masked tokens trained |

---

### Why CLM dominates now

BERT was dominant 2019–2022. GPT-style models have taken over because:

1. **Emergent capabilities** — very large CLM models (100B+) develop reasoning, coding, and instruction-following that MLM models don't
2. **Generative** — one model does everything: chat, code, summarise, classify (with prompting)
3. **Scaling laws** — CLM scales more predictably with data and parameters
4. **Fine-tuning** — SFT + LoRA + RLHF pipelines are built around CLM

BERT-style models are still used for **embeddings** (RAG retrieval) and **fast classification** tasks where you don't need generation.

---

### In BroCode

- **Qwen2.5-0.5B (Causal LM)** — generates Sid's and Guru's replies
- If we added semantic search in Phase 4 using BGE or sentence-transformers, those embedding models would be **MLM-style** — used purely for converting text to vectors, never for generation

---

# EcomBot SFTConfig — Every Parameter Explained

This is the exact config used in `scripts/train_lora.py` with the reasoning behind each value.

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    # WHERE to save checkpoints and the final adapter weights.
    # SFTTrainer saves automatically at save_steps AND at the end.
    # For ecom-bot: 'adapters/ecom-lora'

    num_train_epochs=3,
    # HOW MANY complete passes over the entire dataset.
    # 1 epoch = model sees every training example once.
    # With ~385 examples: 3 epochs = 1,155 training examples seen total.
    # Why 3? Small dataset → needs more passes. Too many → overfitting
    # (bot starts repeating exact training phrases instead of generalising).

    per_device_train_batch_size=4,
    # HOW MANY examples processed in one forward+backward pass.
    # 4 examples → forward pass → loss computed → backward pass.
    # Higher = faster training BUT needs more RAM.
    # 4 is safe for CPU (each example ~512 tokens × float32 = ~2MB).

    gradient_accumulation_steps=2,
    # SIMULATE a larger batch without using more RAM.
    # Instead of updating weights every 4 examples, wait 2 steps then update.
    # Effective batch size = per_device_train_batch_size × accumulation_steps
    #                      = 4 × 2 = 8
    # Why not just set batch_size=8? That needs 2× the RAM.
    # Total steps = ceil(385 examples / 8 per step) × 3 epochs = ~145 steps.

    learning_rate=2e-4,
    # HOW LARGE each weight update step is.
    # LoRA adapters start from: A=random noise, B=zeros.
    # They're learning from scratch → need a HIGHER LR than full fine-tuning.
    # Full fine-tuning: 1e-5 (weights already good, small nudges needed).
    # LoRA:             2e-4 (new weights, must learn quickly).

    warmup_ratio=0.1,
    # GRADUALLY increase LR at the start before reaching 2e-4.
    # First 10% of total steps: LR ramps linearly 0 → 2e-4.
    # With ~145 total steps: warmup = first 14 steps.
    # Why? Prevents huge unstable updates at step 1 when the optimizer
    # has no momentum history and the gradients are noisy.

    lr_scheduler_type="cosine",
    # HOW the LR changes after warmup ends.
    # 'cosine': smooth curve from 2e-4 gradually toward ~0 by the last step.
    # Better than 'linear' (drops too steeply early) or 'constant' (never decays).
    # Cosine gives full speed early and slows down gently at the end.

    bf16=False,
    # BFloat16 — NOT supported on CPU or older GPUs (needs Ampere: A100, H100).
    # BF16 keeps the same number range as FP32 but with less precision.
    # We trained on CPU so this must be False.

    fp16=False,
    # Float16 — halves memory vs FP32, faster on GPU.
    # Also False because we're on CPU. FP16 on CPU has no speedup
    # and can cause numerical instability.
    # On a CUDA GPU you'd set: fp16 = (DEVICE != 'cpu')

    logging_steps=10,
    # PRINT training metrics every N optimizer steps.
    # Every 10 steps you see: loss, learning_rate, epoch.
    # Lets you confirm loss is going down: 4.3 → 2.1 → 1.4 → 1.0 → ...
    # With ~145 total steps this gives ~14 log prints.

    save_steps=100,
    # SAVE a checkpoint every N steps.
    # checkpoint-100 = weights at step 100 (safety net if training crashes).
    # With ~145 steps: saves once mid-training (checkpoint-100)
    # and once at the end (checkpoint-145 + the root adapter).
    # Use root 'adapters/ecom-lora/' for inference, not checkpoints.

    max_length=512,
    # MAXIMUM token length for each training example.
    # Examples longer than 512 tokens are truncated.
    # E-commerce prompts are short (< 200 tokens), so this rarely triggers.
    # Prevents OOM on unexpectedly long examples.
    # Note: some SFTConfig versions call this max_seq_length — same thing.

    report_to="none",
    # WHERE to send training metrics.
    # Default tries to log to wandb (Weights & Biases) or TensorBoard.
    # 'none' = just print to terminal. No external service needed.
)
```

## What the training logs mean

When training runs, every `logging_steps` you see:

```
{'loss': '4.379', 'grad_norm': '5.747', 'learning_rate': '0.0001999', 'epoch': '0.4'}
{'loss': '2.212', 'grad_norm': '3.082', 'learning_rate': '0.000187',  'epoch': '0.8'}
{'loss': '1.459', 'grad_norm': '1.791', 'learning_rate': '0.0001553', 'epoch': '1.2'}
{'loss': '1.029', 'grad_norm': '1.181', 'learning_rate': '7.775e-05', 'epoch': '1.905'}
{'loss': '0.939', 'grad_norm': '1.254', 'learning_rate': '2.507e-06', 'epoch': '2.857'}
```

| Field | What it means | Healthy sign |
|-------|--------------|--------------|
| `loss` | How wrong the model is. Cross-entropy on next-token prediction | Decreasing each epoch |
| `grad_norm` | Size of the gradient vector. Spikes mean unstable training | Stays below ~5 after warmup |
| `learning_rate` | Current LR — shows the cosine decay happening | Starts high, ends near 0 |
| `epoch` | How far through the dataset you are | Goes 0 → 3.0 |

**EcomBot final loss ~0.94** — that's well-converged for a 0.5B model on 385 examples.

