import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from datasets import load_dataset
import bitsandbytes as bnb
import torch

# ── 1.  Model & tokenizer ───────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # needed for padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    use_cache=False,
)
model.gradient_checkpointing_enable()      # saves VRAM

# ── 2.  Dataset ─────────────────────────────────────────────────────
dataset = load_dataset("json", data_files="galactus_dataset.json")

def tokenize(batch):
    merged = [
        f"{i} {j} {k}"
        for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    return tokenizer(
        merged,
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenised = dataset["train"].map(tokenize, batched=True)

# ── 3.  Training arguments  ─────────────────────────────────────────
train_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,   # effective batch = 2
    num_train_epochs=5,              # bump as you add more data
    fp16=False,
    max_grad_norm=0.0,               # disables grad-clipping scaler
    logging_steps=10,

    save_strategy="no",              # no mid-epoch checkpoints
    save_total_limit=1,
    overwrite_output_dir=True,
    report_to="none",
)

# ── 4.  Low-VRAM optimiser  ─────────────────────────────────────────
optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=5e-5)

# ── 5.  Trainer  ────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenised,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    optimizers=(optimizer, None)     # custom optimiser, no scheduler
)

# ── 6.  Train & save  ───────────────────────────────────────────────
trainer.train()
trainer.save_model("./galactus2-model")     # final lightweight checkpoint
