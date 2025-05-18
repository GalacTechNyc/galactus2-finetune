from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)
model.gradient_checkpointing_enable()  # 🔥 Less memory use

dataset = load_dataset("json", data_files="galactus_dataset.json")

def tokenize(batch):
    combined = [i + " " + j + " " + k for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    fp16=False,
    save_steps=10,
    logging_steps=10,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
