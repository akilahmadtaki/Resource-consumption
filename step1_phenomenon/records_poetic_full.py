import sys, os, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.jsonl")
records = [json.loads(l) for l in open(PATH)]
easy = [r for r in records if r["subset"] == "logical_deduction_three_objects"]
print(f"Running {len(easy)} records x 3 conditions\n")

CONDS = ["plain_original", "plain_matched_poetic", "poetic_rich"]
print(f"{'item':>6} {'plain_orig':>11} {'plain_match':>12} {'poetic':>8} {'style_ratio':>12}")
print("-" * 56)

style_ratios, noise_ratios, loops = [], [], 0
for r in easy:
    vals = {}
    for c in CONDS:
        vals[c], _ = measure_reasoning(r["conditions"][c], max_new_tokens=6000)

    # paired ratios (per record, not averaged across records)
    sr = nr = None
    if vals["poetic_rich"] and vals["plain_matched_poetic"]:
        sr = vals["poetic_rich"] / vals["plain_matched_poetic"]
        style_ratios.append(sr)
    if vals["plain_matched_poetic"] and vals["plain_original"]:
        nr = vals["plain_matched_poetic"] / vals["plain_original"]
        noise_ratios.append(nr)
    loops += sum(1 for v in vals.values() if v is None)

    print(f"{r['item_id'][-4:]:>6} {str(vals['plain_original']):>11} "
          f"{str(vals['plain_matched_poetic']):>12} {str(vals['poetic_rich']):>8} "
          f"{(f'{sr:.2f}x' if sr else '-'):>12}")

print("-" * 56)
print(f"\nLoops/failures: {loops}")
print("\nPAIRED RATIOS (per-record, then averaged):")
print(f"  1. STYLE (length controlled)  mean {statistics.mean(style_ratios):.2f}x  "
      f"median {statistics.median(style_ratios):.2f}x  (n={len(style_ratios)})")
print(f"  2. NOISE FLOOR (rewording)    mean {statistics.mean(noise_ratios):.2f}x  "
      f"median {statistics.median(noise_ratios):.2f}x  (n={len(noise_ratios)})")
print(f"\n  style ratio spread: min {min(style_ratios):.2f}x  max {max(style_ratios):.2f}x")
