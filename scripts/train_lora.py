"""
Train EcomBot LoRA adapter on Qwen2.5-0.5B-Instruct.
Run: python scripts/train_lora.py
"""
from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "data/train.jsonl"
OUTPUT_DIR   = "adapters/ecom-lora"
DEVICE       = "cpu"   # MPS hangs on weight loading; switch to mps/cuda once confirmed working

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map=DEVICE,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Dataset: {len(dataset)} examples")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # effective batch = 8
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=False,
    fp16=False,   # CPU only
    logging_steps=10,
    save_steps=100,
    max_length=512,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print(f"\nTraining on {DEVICE}...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved → {OUTPUT_DIR}/")
