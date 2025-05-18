from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Load Phi-2 (placeholder for Phi-3 when available)
model_name = "microsoft/phi-2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix: Set pad token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Load dataset from JSON
dataset = load_dataset("json", data_files="galactus_dataset.json")

# Tokenization logic
def tokenize(batch):
    combined = [i + " " + j + " " + k for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=512)

# Tokenize the dataset
tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=True,
    save_steps=10,
    logging_steps=10,
)

# Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the model
trainer.train()
