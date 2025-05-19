# train_lora.py  –  Rapid LoRA fine-tune for Galactus-2
# -----------------------------------------------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ── settings ─────────────────────────────────────────
BASE_MODEL = "microsoft/phi-2"       # swap to phi-1_5 for faster drafts
DATA_PATH  = "galactus_dataset.json"
OUTPUT_DIR = "galactus2-lora"
EPOCHS     = 3
LR         = 1e-4

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model
)

# ── tokenizer ───────────────────────────────────────
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.pad_token = tok.eos_token

# ── base model load ──────────────────────────────────
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# ── LoRA adapter config ─────────────────────────────
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)

# ── dataset ─────────────────────────────────────────
def tokenize(batch):
    merged = [
        f"{i} {j} {k}"
        for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    return tok(merged, truncation=True, padding="max_length", max_length=256)

ds = load_dataset("json", data_files=DATA_PATH)["train"].map(tokenize, batched=True)

# ── training arguments ──────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    save_strategy="no",            # only final adapter gets saved
    logging_steps=10,
    report_to="none"
)

# ── train LoRA adapter ──────────────────────────────
Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
).train()

# ── save adapter ────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
print(f"🎉 LoRA adapter saved to {OUTPUT_DIR}/")
