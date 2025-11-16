from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import os
import random
import pyarrow.parquet as pq

print("Downloading HumanEval...")
# Use HuggingFace version: openai/openai_humaneval
# Download Parquet file directly to avoid fsspec issues with datasets==2.14.0
print("  Downloading Parquet file directly...")
parquet_file = hf_hub_download(
    repo_id="openai/openai_humaneval",
    filename="openai_humaneval/test-00000-of-00001.parquet",
    repo_type="dataset"
)

# Read Parquet file
table = pq.read_table(parquet_file)
humaneval_data = []
for i in range(len(table)):
    row = {}
    for col in table.column_names:
        row[col] = table[col][i].as_py()
    humaneval_data.append(row)

print(f"HumanEval total: {len(humaneval_data)}")

# Sample 50
random.seed(42)
humaneval_sample = random.sample(humaneval_data, min(50, len(humaneval_data)))

print(f"HumanEval sampled: {len(humaneval_sample)}")

# Ensure directory exists (remove if it's a file)
humaneval_dir = "data/raw/humaneval"
if os.path.exists(humaneval_dir) and not os.path.isdir(humaneval_dir):
    os.remove(humaneval_dir)
os.makedirs(humaneval_dir, exist_ok=True)
with open("data/raw/humaneval/samples.json", "w") as f:
    json.dump(humaneval_sample, f, indent=2)

print("Downloading HelpSteer2...")
# Try normal load first, fallback to workaround if needed
try:
    helpsteer2 = load_dataset("nvidia/HelpSteer2")
    help_data = [dict(item) for item in helpsteer2['train']]
except (NotImplementedError, ValueError) as e:
    if "LocalFileSystem" in str(e) or "Invalid pattern" in str(e):
        # Use workaround: download JSONL file directly (HelpSteer2 uses JSONL, not Parquet)
        print("  Using workaround: downloading JSONL file directly...")
        import gzip
        jsonl_file = hf_hub_download(
            repo_id="nvidia/HelpSteer2",
            filename="train.jsonl.gz",
            repo_type="dataset"
        )
        
        # Read and parse JSONL file
        help_data = []
        with gzip.open(jsonl_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    help_data.append(json.loads(line))
    else:
        raise

print(f"HelpSteer2 total: {len(help_data)}")

# Ensure directory exists (remove if it's a file)
helpsteer2_dir = "data/raw/helpsteer2"
if os.path.exists(helpsteer2_dir) and not os.path.isdir(helpsteer2_dir):
    os.remove(helpsteer2_dir)
os.makedirs(helpsteer2_dir, exist_ok=True)
with open("data/raw/helpsteer2/full.json", "w") as f:
    json.dump(help_data, f, indent=2)

print("HumanEval and HelpSteer2 saved")