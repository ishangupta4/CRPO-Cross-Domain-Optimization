from data_loader import DataLoader
import json
import os

loader = DataLoader()

print("=" * 50)
print("PREPROCESSING ALL DATASETS")
print("=" * 50)

os.makedirs("data/processed", exist_ok=True)

# GSM8K
print("\n1. GSM8K...")
gsm8k_train = loader.load_gsm8k('train')
gsm8k_test = loader.load_gsm8k('test')
with open("data/processed/gsm8k_train.json", "w") as f:
    json.dump(gsm8k_train, f)
with open("data/processed/gsm8k_test.json", "w") as f:
    json.dump(gsm8k_test, f)
print(f"   Train: {len(gsm8k_train)}, Test: {len(gsm8k_test)}")

# BBH Navigate
print("\n2. BBH Navigate...")
bbh_nav_train, bbh_nav_test = loader.load_bbh('navigate')
with open("data/processed/bbh_navigate_train.json", "w") as f:
    json.dump(bbh_nav_train, f)
with open("data/processed/bbh_navigate_test.json", "w") as f:
    json.dump(bbh_nav_test, f)
print(f"   Train: {len(bbh_nav_train)}, Test: {len(bbh_nav_test)}")

# BBH Boolean
print("\n3. BBH Boolean...")
bbh_bool_train, bbh_bool_test = loader.load_bbh('boolean_expressions')
with open("data/processed/bbh_boolean_train.json", "w") as f:
    json.dump(bbh_bool_train, f)
with open("data/processed/bbh_boolean_test.json", "w") as f:
    json.dump(bbh_bool_test, f)
print(f"   Train: {len(bbh_bool_train)}, Test: {len(bbh_bool_test)}")

# LIAR
print("\n4. LIAR...")
liar_opt = loader.load_liar('optimization')
liar_test = loader.load_liar('test')
with open("data/processed/liar_optimization.json", "w") as f:
    json.dump(liar_opt, f)
with open("data/processed/liar_test.json", "w") as f:
    json.dump(liar_test, f)
print(f"   Optimization: {len(liar_opt)}, Test: {len(liar_test)}")

# HumanEval
print("\n5. HumanEval...")
code = loader.load_humaneval(50)
code_train = code[:25]
code_test = code[25:]
with open("data/processed/code_train.json", "w") as f:
    json.dump(code_train, f)
with open("data/processed/code_test.json", "w") as f:
    json.dump(code_test, f)
print(f"   Train: {len(code_train)}, Test: {len(code_test)}")

# HelpSteer2
print("\n6. HelpSteer2...")
helpsteer2 = loader.load_helpsteer2()
with open("data/processed/helpsteer2_full.json", "w") as f:
    json.dump(helpsteer2, f)
print(f"   Total: {len(helpsteer2)}")

print("\n" + "=" * 50)
print("ALL DATASETS PREPROCESSED SUCCESSFULLY")