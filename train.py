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
import torch
import bitsandbytes as bnb

# Model selection
model_name = "microsoft/phi-2"

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Model loading with memory optimization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_cache=False
)
model.gradient_checkpointing_enable()

# Load dataset
dataset = load_dataset("json", data_files="galactus_dataset.json")

# Tokenize inputs
def tokenize(batch):
    combined = [i + " " + j + " " + k for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Training settings
training_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=10,
    overwrite_output_dir=True,
    report_to="none"
)

# 💡 Optimizer: 32-bit low-memory AdamW
optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=5e-5)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    optimizers=(optimizer, None)
)

# 🚀 Train Galactus2
trainer.train()
