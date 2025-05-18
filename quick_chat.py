#!/usr/bin/env python3
"""
quick_chat.py  —  Galactus-2 demo with LoRA adapter
---------------------------------------------------
• 8-bit base model load (fast)
• Post-processing to remove echoed prompt
• Interactive REPL: type a question, get an answer
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, readline   # readline = nicer CLI

BASE_MODEL   = "microsoft/phi-2"
ADAPTER_PATH = "galactus2-lora"          # folder from train_lora.py
DEVICE_MAP   = "auto"

print("⏳ Loading base…")
quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
tok  = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_cfg,
    device_map=DEVICE_MAP
)
print("⏳ Loading LoRA adapter…")
model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()

def chat(user_prompt: str) -> str:
    """Generate a Galactus-style answer without echoing the prompt."""
    prompt = f"User: {user_prompt.strip()}\nAssistant:"
    
    out = model.generate(
        **tok(prompt, return_tensors="pt").to("cuda"),
        max_new_tokens=80,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.3,
        eos_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # Strip everything before Assistant:
    answer = text.split("Assistant:", 1)[-1]
    # Cut off if model starts a new user turn
    answer = answer.split("\nUser:", 1)[0]

    return answer.strip()

# ── REPL ────────────────────────────────────────────
print("🎙  Galactus-2 ready. Type your question (or 'exit'):\n")
while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    if not q or q.lower() in {"exit", "quit"}:
        print("Bye!")
        break
    print("Galactus-2:", chat(q), "\n")
