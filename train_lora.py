from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import bitsandbytes as bnb

BASE = "microsoft/phi-2"          # switch to phi-1_5 for faster prototyping
DS_PATH = "galactus_dataset.json"

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

# 8-bit load to save VRAM
base_model = AutoModelForCausalLM.from_pretrained(BASE, load_in_8bit=True, device_map="auto")
base_model = prepare_model_for_kbit_training(base_model)

# LoRA config (tiny adapter)
lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora)

def tokenize(batch):
    merged = [f"{i} {j} {k}" for i, j, k in zip(batch["instruction"], batch["input"], batch["output"])]
    return tok(merged, truncation=True, padding="max_length", max_length=256)

ds = load_dataset("json", data_files=DS_PATH)["train"].map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="galactus2-lora",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
).train()

model.save_pretrained("galactus2-lora")
print("🎉 LoRA adapter saved to galactus2-lora/")
