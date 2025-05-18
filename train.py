from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

dataset = load_dataset("json", data_files="galactus_dataset.json")

def tokenize(batch):
    combined = [i + " " + j + " " + k for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./galactus2-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=True,
    save_steps=10,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

