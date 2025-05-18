import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Choose model
model_name = "microsoft/phi-2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # optionally "auto"
    use_cache=False
)

model.gradient_checkpointing_enable()

# Load dataset
dataset = load_dataset("json", data_files="galactus_dataset.json")

# Tokenize
def tokenize(batch):
    combined = [i + " " + j + " " + k for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=10,
    logging_steps=10,
    bf16=True,  # ✔️ safer than fp16 on modern GPUs
    fp16=False, # 🔥 turn off fp16 to avoid unscale error
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
