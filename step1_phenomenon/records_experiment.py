import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning
import statistics

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.jsonl")
records = [json.loads(l) for l in open(PATH)]
easy = [r for r in records if r["subset"] == "logical_deduction_three_objects"]

N = 5   # start small
CONDS = ["plain_original", "plain_matched_poetic", "poetic_rich"]

print(f"{'item':>6} " + " ".join(f"{c[:18]:>18}" for c in CONDS))
print("-" * 64)

results = {c: [] for c in CONDS}
for r in easy[:N]:
    row = []
    for c in CONDS:
        n, _ = measure_reasoning(r["conditions"][c], max_new_tokens=6000)
        results[c].append(n)
        row.append(str(n))
    print(f"{r['item_id'][-4:]:>6} " + " ".join(f"{v:>18}" for v in row))

print("-" * 64)
print("\nMEANS (skipping any None):")
means = {}
for c in CONDS:
    vals = [v for v in results[c] if v is not None]
    means[c] = statistics.mean(vals) if vals else None
    print(f"  {c:<24} {means[c]:.0f} tokens  (n={len(vals)})")

print("\nTHE TWO COMPARISONS:")
if means["plain_matched_poetic"] and means["poetic_rich"]:
    r1 = means["poetic_rich"] / means["plain_matched_poetic"]
    print(f"  1. STYLE (length controlled): poetic_rich / plain_matched = {r1:.2f}x")
if means["plain_original"] and means["plain_matched_poetic"]:
    r2 = means["plain_matched_poetic"] / means["plain_original"]
    print(f"  2. NOISE FLOOR (rewording only): plain_matched / plain_original = {r2:.2f}x")
