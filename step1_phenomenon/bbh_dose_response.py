import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import measure_reasoning
from bbh_styler import style_puzzle, STYLERS
from datasets import load_dataset
import numpy as np

# low-baseline puzzles (most headroom for inflation)
PUZZLES = [26, 31, 63, 48, 81, 17]

d = load_dataset("lukaemon/bbh", "logical_deduction_five_objects")["test"]
styles = ["archaic_light", "archaic_heavy", "song_light", "song_heavy"]

print(f"{'puzzle':>6} {'plain':>6} " + " ".join(f"{s:>14}" for s in styles))
print("-" * 78)

ratios = {s: [] for s in styles}
for i in PUZZLES:
    plain_text = d[i]["input"]
    base, _ = measure_reasoning(plain_text, max_new_tokens=6000)
    if base is None:
        print(f"{i:>6} {'LOOP':>6}  (plain looped — skip)")
        continue

    cells = []
    for s in styles:
        n, _ = measure_reasoning(style_puzzle(plain_text, s), max_new_tokens=6000)
        if n is None:
            cells.append("None")
        else:
            r = n / base
            ratios[s].append(r)
            cells.append(f"{n} ({r:.1f}x)")
    print(f"{i:>6} {base:>6} " + " ".join(f"{c:>14}" for c in cells))

print("-" * 78)
print("\nMEAN INFLATION BY STYLE (dose-response):")
for s in styles:
    if ratios[s]:
        print(f"  {s:>14}: {np.mean(ratios[s]):.2f}x   (n={len(ratios[s])})")

print("\nLIGHT vs HEAVY:")
for base_style in ["archaic", "song"]:
    l = ratios.get(f"{base_style}_light", [])
    h = ratios.get(f"{base_style}_heavy", [])
    if l and h:
        print(f"  {base_style}: light {np.mean(l):.2f}x  ->  heavy {np.mean(h):.2f}x")
