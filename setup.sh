#!/bin/bash
apt update && apt install -y git python3-pip
pip install torch transformers datasets peft accelerate
echo "✅ Environment setup complete."
