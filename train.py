import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from datasets import load_dataset
import torch, bitsandbytes as bnb

# ── 1.  Model & tokenizer ───────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # full precision keeps things stable
    use_cache=False
)
model.gradient_checkpointing_enable()     # save VRAM

# ── 2.  Dataset ─────────────────────────────────────────────────────
ds = load_dataset("json", data_files="galactus_dataset.json")

def tok(batch):
    merged = [
        f"{i} {j} {k}"
        for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    return tokenizer(merged, truncation=True, padding="max_length", max_length=256)

tok_ds = ds["train"].map(tok, batched=True)

# ── 3.  Training arguments  ─────────────────────────────────────────
train_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,   # effective batch = 2
    num_train_epochs=3,
    fp16=False,                      # avoid AMP scaler issues
    max_grad_norm=0.0,               # disables grad-clipping scaler
    logging_steps=10,

    save_strategy="no",              # 🚫 no mid-run checkpoints
    save_total_limit=1,
    save_optimizer_state=False,      # 🚫 skip huge optimiser states

    overwrite_output_dir=True,
    report_to="none"
)

# ── 4.  Low-VRAM optimiser  ────────────────────────────────────────
optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=5e-5)

# ── 5.  Trainer  ───────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tok_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    optimizers=(optimizer, None)
)

# ── 6.  Train & manual save  ───────────────────────────────────────
trainer.train()
trainer.save_model("./galactus2-model")   # ← final tiny checkpoint only
