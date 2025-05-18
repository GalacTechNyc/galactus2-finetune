from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Load base model (use phi-2 for now, upgrade later to phi-3 if weights drop)
model_name = "microsoft/phi-2"

# Tokenizer & model loading
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix: required pad token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Load and format dataset
dataset = load_dataset("json", data_files="galactus_dataset.json")

# Batch-safe tokenization
def tokenize(batch):
    combined = [i + " " + j + " " + k for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=512)

# Tokenize the dataset
tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=False,  # Disable mixed precision to avoid CUDA error
    save_steps=10,
    logging_steps=10,
)

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train!
trainer.train()
