import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.jsonl")

# JSONL: one complete JSON object per line
records = [json.loads(line) for line in open(PATH)]
print(f"Loaded {len(records)} records")

# keep only the easiest subset (lowest baselines = most headroom for the effect)
easy = [r for r in records if r["subset"] == "logical_deduction_three_objects"]
print(f"logical_deduction_three_objects: {len(easy)} records")

# inspect the three conditions we will compare, on the first record
r = easy[0]
print(f"\nExample record: {r['item_id']}  (answer {r['target']})")
for c in ["plain_original", "plain_matched_poetic", "poetic_rich"]:
    text = r["conditions"][c]
    print(f"  {c:<24} {len(text.split()):>4} words")
