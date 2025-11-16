from datasets import load_dataset
import json
import os

print("Downloading GSM8K...")
gsm8k = load_dataset("openai/gsm8k", "main")

# Save raw data
os.makedirs("data/raw/gsm8k", exist_ok=True)

train_data = gsm8k['train']
test_data = gsm8k['test']

print(f"Train examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

# Save as JSON
with open("data/raw/gsm8k/train.json", "w") as f:
    json.dump([dict(item) for item in train_data], f)

with open("data/raw/gsm8k/test.json", "w") as f:
    json.dump([dict(item) for item in test_data], f)

print("GSM8K saved to data/raw/gsm8k/")
